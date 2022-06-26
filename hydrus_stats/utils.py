import json
from typing import Any, Dict, List, Tuple


def id_to_url(id, url, api_key) -> str:
    """
    Convert a file ID to a URL.
    :param id: File ID.
    :param url: URL of the API.
    :param api_key: API key.
    :return: URL of the file.
    """
    return (
        url
        + "get_files/file?file_id="
        + str(id)
        + "&Hydrus-Client-API-Access-Key="
        + api_key
    )


def ids_to_url(file_ids) -> List[str]:
    return [id_to_url(id) for id in file_ids]


def get_tags(entry: Dict[str, Any], exclude_namespaced=False) -> List[str]:
    """
    Get tags from entry.
    :param entry: Metadata dictionary.
    :param exclude_namespaced: Whether to exclude namespaced tags.
    :return: List of tags.
    """
    result = set()
    display_tags = entry["service_names_to_statuses_to_display_tags"]
    for service in display_tags:
        try:
            tags = display_tags[service]["0"]
        except KeyError:
            continue
        for t in tags:
            if exclude_namespaced:
                if len(t.split(":")) > 1:
                    continue
            result.add(t)

    return list(result)


def load_counts_from_file(filename: str) -> Tuple[Dict[str, int], int, int]:
    """Load tag counts from file.
    :param filename: File to load from.
    :return: Dictionary of tag counts, number of documents, number of tags.
    """
    with open(filename, encoding="utf8") as f:
        data_dict: Dict[str, int] = json.load(f)

    return data_dict["counts"], data_dict["num_documents"], data_dict["num_tags"]