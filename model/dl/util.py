"""
This is a utility module storing helper or non-core function for `glove.py`.
"""

from json import load


def json_to_text():
    """ Convert jsoned json string to purely json."""
    with open('my_model.json', 'r') as fp:
        string = load(fp)
        with open('my_model_json.txt', 'w') as out:
            out.write(string)
