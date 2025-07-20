from graph2.query_route import question_route_chain

res = question_route_chain.invoke({'question':'Milvus on Windows'})
print(type(res.datasource))