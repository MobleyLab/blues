def write_version_py(filename='meta.yaml'):
    cnt = """
short_version = '{short_version:s}'
build_number = '{build_number:s}'
version = '{version:s}'
full_version = '{full_version:s}'
git_revision = '{git_revision:s}'
release = {release:s}
"""
    d = {}
    with open('../../blues/version.py') as f:
        data = f.read()
    lines = data.split('\n')

    for line in lines:
        keys = ['version', 'build_number', 'git_revision', 'release']
        for k in keys:
            if k in line:
                (key, val) = line.split('=')
                d[key.strip()] = val.strip().strip("'")

    b = open('meta.yaml', 'r')
    yaml_lines = b.read()

    a = open(filename, 'w')
    try:
        for k,v in d.items():
            a.write("{{% set {} = '{}' %}}\n".format(k,v))
        a.write(yaml_lines)
    finally:
        a.close()
write_version_py('meta.yaml')
