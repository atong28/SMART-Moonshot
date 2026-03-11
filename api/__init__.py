from .server import ServerAPI


def get_api() -> ServerAPI: 
    api = ServerAPI()
    api.initialize()
    return api


__all__ = ["ServerAPI", "get_api"]
