from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
import datetime

Base = declarative_base()

class UserRequest(Base):
    __tablename__ = "user_requests"
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    agent_type = Column(String)
    question = Column(String)
    response = Column(String)
