def lisp_to_label(lisp: str, mid_to_node: dict):
    lisp_split = lisp.split(' ')
    res = []
    for item in lisp_split:
        r_par_count = item.count(')')
        item = item.rstrip(')')
        r_par = r_par_count > 0

        inv = False
        if item.endswith('_inv'):
            inv = True
            item = item.replace('_inv', '')

        if item.startswith('https://github.com/OSU-NLP-Group/HippoRAG/') and 'Node' in item:
            res.append('[' + mid_to_node[item].replace('https://github.com/OSU-NLP-Group/HippoRAG', '') + ']')
        elif item.startswith('https://github.com/OSU-NLP-Group/HippoRAG/'):
            label = item.replace('https://github.com/OSU-NLP-Group/HippoRAG', '')
            label = label.split('/')[-1].replace('_', ' ')
            res.append('[' + label + ']')
        else:
            res.append(item)

        if inv:
            res[-1] += '_inv'

        if r_par:
            res[-1] += ')' * r_par_count

    return ' '.join(res)


def simplify_with_prefixes(lisp: str, prefixes: dict):
    for prefix in prefixes:
        full = prefixes[prefix]
        lisp = lisp.replace(full, f"{prefix}:")
    return lisp


def replace_inv_relations(lisp: str, inv_label: dict):
    for inv in inv_label:
        lisp = lisp.replace(inv, inv_label[inv])
    return lisp


def lisp_to_repr(lisp, mid_to_node, prefixes=None, inv_label=None):
    lisp = lisp_to_label(lisp, mid_to_node)
    if prefixes:
        lisp = simplify_with_prefixes(lisp, prefixes)
    if inv_label:
        lisp = replace_inv_relations(lisp, inv_label)
    return lisp


def execute_query_with_virtuoso(query: str, endpoint='http://localhost:3003/sparql'):
    """
    Using SPARQLWrapper to execute the query
    :param query:
    :param endpoint:
    :return:
    """
    from SPARQLWrapper import SPARQLWrapper, JSON
    sparql = SPARQLWrapper(endpoint)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    response = sparql.query().convert()
    rows = []
    for result in response["results"]["bindings"]:
        cur_row = []
        skip = False  # skip the results from openlinksw
        for key in result:
            if result[key]["value"].startswith('http://www.openlinksw.com/'):
                skip = True
                break
            cur_row.append(result[key]["value"])
        if not skip:
            rows.append(tuple(cur_row))
    return rows


if __name__ == '__main__':
    sparql = 'SELECT ?s WHERE { ?s ?p ?o } LIMIT 10000'
    results = execute_query_with_virtuoso(sparql)
    print(results)
    print(len(results))
