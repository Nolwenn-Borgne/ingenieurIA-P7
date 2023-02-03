# -*- coding: utf-8 -*-

from pydantic import BaseModel

# 1. Class which describes Bank Notes measurements
class Tweet(BaseModel):
    text: str
