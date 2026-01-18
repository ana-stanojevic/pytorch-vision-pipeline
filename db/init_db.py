from .base import Base
from .session import engine
from .models import Prediction  

def init_db():
    Base.metadata.create_all(bind=engine)