from sqlalchemy import Column, BigInteger, Text, Float, TIMESTAMP, func
from .base import Base

class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    sha256 = Column(Text, nullable=False, unique=True, index=True)
    class_name = Column (Text)
    tag = Column (Text)
    confidence = Column (Float, nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)