from log.log_utils import log


def draw_graph(graph, file_path: str):
    try:
        image = graph.get_graph().draw_mermaid_png()
        with open(file_path, 'wb') as f:
            f.write(image)
    except Exception as e:
        log.exception(e)