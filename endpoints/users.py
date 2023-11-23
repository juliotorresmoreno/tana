from fastapi import APIRouter
from sqlalchemy import select
from database import SessionLocal, engine
from models.users import User

session = SessionLocal(engine)

router = APIRouter()

@router.get("/users")
async def ask():
    stmt = select(User).where(User.name.in_(["spongebob", "sandy"]))
    sandy_address = session.scalars(stmt).all()

    return {"answer": prompt, "response": response, "execution_time": execution_time}