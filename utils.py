import json


def idToURL(id, url, api_key):
    return (
        url
        + "get_files/file?file_id="
        + str(id)
        + "&Hydrus-Client-API-Access-Key="
        + api_key
    )


def idsToURL(file_ids):
    return [idToURL(id) for id in file_ids]


def get_tags(entry, exclude_namespaced=False):
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


def load_counts_from_file(filename):

    with open(filename) as f:
        data_dict = json.load(f)

    return data_dict["counts"], data_dict["num_documents"], data_dict["num_tags"]