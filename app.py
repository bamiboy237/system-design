import json
import time
from contextlib import asynccontextmanager
from enum import Enum
from typing import Annotated, Literal
from uuid import uuid4

import redis.asyncio as redis
from fastapi import (
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Request,
    Response,
    status,
)
from pydantic import BaseModel, Field

TOKEN_BUCKET_LUA = """
local key = KEYS[1]
local capacity = tonumber(ARGV[1])
local refill_rate = tonumber(ARGV[2])
local cost = tonumber(ARGV[3])
local now = tonumber(ARGV[4])

-- 1. Fetch current bucket state
local data = redis.call("HMGET", key, "tokens", "last_updated")
local tokens = tonumber(data[1])
local last_updated = tonumber(data[2])

-- 2. Initialize or Refill
if tokens == nil or last_updated == nil then
    tokens = capacity
    last_updated = now
else
    local elapsed = math.max(0, now - last_updated)
    tokens = math.min(capacity, tokens + (elapsed * refill_rate))
    last_updated = now
end

-- 3. Check and Consume
if tokens >= cost then
    tokens = tokens - cost
    redis.call("HSET", key, "tokens", tokens, "last_updated", last_updated)
    -- Expire after key would have completely refilled + buffer (e.g. 60s) to reclaim RAM
    redis.call("EXPIRE", key, 60)
    return 1 -- Allowed
else
    redis.call("HSET", key, "tokens", tokens, "last_updated", last_updated)
    redis.call("EXPIRE", key, 60)
    return 0 -- Denied
end

"""


class Status(str, Enum):
    IN_PROGRESS = "IN PROGRESS"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


class ChargeRequest(BaseModel):
    amount: int = Field(gt=0, description="Amount in cents")
    currency: Literal["USD", "EUR", "GBP"] = Field(
        default="USD", description="Charge Currency"
    )


class ChargeResponse(BaseModel):
    charge_id: str
    amount: int
    currency: Literal["USD", "EUR", "GBP"]
    status: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.redis = redis.from_url("redis://localhost:6379", decode_responses=True)
    yield
    await app.state.redis.aclose()


app = FastAPI(title="Payment Gateway", lifespan=lifespan)


async def get_redis(request: Request) -> redis.Redis:
    return request.app.state.redis


async def is_rate_limited(
    redis_client: redis.Redis,
    client_id: str,
    capacity: int = 10,
    refill_rate: float = 2.0,
    cost: int = 1,
) -> bool:
    """
    Returns True if rate limited (rejected), False if allowed.
    """
    now = time.time()
    result = await redis_client.eval(
        TOKEN_BUCKET_LUA, 1, f"ratelimit:{client_id}", capacity, refill_rate, cost, now
    )
    return result == 0


@app.post("/v1/charges")
async def process_charge(
    req: ChargeRequest,
    response: Response,
    x_client_id: Annotated[str, Header()],
    idempotency_key: Annotated[str, Header(alias="Idempotency-key")],
    redis=Depends(get_redis),
):
    if await is_rate_limited(redis, x_client_id, capacity=10, refill_rate=2.0):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Rate Limit Exceeded"
        )
    idempotency_status, cached_response = await check_idempotency(
        redis, idempotency_key
    )
    if idempotency_status == "COMPLETED":
        response.headers["X-Cache"] = "HIT"
        return ChargeResponse.model_validate(cached_response)
    elif idempotency_status == "IN_PROGRESS":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Request is already in progress",
        )

    payment_response = ChargeResponse(
        charge_id=f"ch_{uuid4().hex[:10]}",
        amount=req.amount,
        currency=req.currency,
        status=Status.SUCCEEDED,
    )

    await complete_idempotency(
        redis,
        idempotency_key,
        payment_response.model_dump(mode="json"),
    )
    response.headers["X-Cache"] = "MISS"
    return payment_response


async def check_idempotency(
    redis_client: redis.Redis,
    idempotency_key: str,
) -> tuple[str, dict | None]:
    key = f"idempotency:{idempotency_key}"

    acquired = await redis_client.set(
        key,
        json.dumps({"status": "IN_PROGRESS"}),
        ex=15,
        nx=True,
    )

    if acquired:
        return ("NEW", None)

    val = await redis_client.get(key)

    if val is None:
        return ("NEW", None)

    data = json.loads(val)

    if data.get("status") == "IN_PROGRESS":
        return ("IN_PROGRESS", None)

    return ("COMPLETED", data.get("response"))


async def complete_idempotency(
    redis_client: redis.Redis, idempotency_key: str, response: dict, ttl: int = 86400
) -> None:
    """Saves the COMPLETED state and response data with a 24 hour TTL."""
    key = f"idempotency:{idempotency_key}"
    await redis_client.set(
        key,
        json.dumps(
            {
                "status": "COMPLETED",
                "response": {
                    "charge_id": response["charge_id"],
                    "amount": response["amount"],
                    "currency": response["currency"],
                    "status": response["status"],
                },
            }
        ),
        ex=ttl,
    )
