# 创建TavilySearThResults工具，设置最大结果数为2
import os
from langchain_tavily import TavilySearch
from utils.env import TAVILY_API_KEY

os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
web_search_tool = TavilySearch(max_results=3)