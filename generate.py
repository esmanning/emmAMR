from amr_hackathon_master.src.amr import * # from https://github.com/nschneid/amr-hackathon
from queue import PriorityQueue
from itertools import permutations
import kenlm
import pickle
import argparse
import operator
import math


def score_order(order):
    """scores a given hypothesized linearization using one or both linearization models"""
    if options.p:
        pair_prob = 0
        for i, first in enumerate(order):
            for j, second in enumerate(order[i:]):
                pair = (first[1], second[1])
                if pair in pair_order_model:
                    pair_prob -= math.log(pair_order_model[pair])
        if not options.c:
            return pair_prob
    if options.c:
        coreness_prob = 0
        before = []
        after = []
        seen_root = False
        for triple in order:
            if triple[1] == 'ROOT':
                seen_root = True
            elif seen_root is False:
                before.insert(0, triple)
            else:
                after.append(triple)
        if len(before) >= 2:
            for i, triple in enumerate(before):
                if triple[1] in coreness_order_model:
                    expected = coreness_order_model[triple[1]]
                    observed = i / (len(before)-1)
                    diff = abs(expected-observed)
                    if diff != 0:
                        coreness_prob -= math.log(diff)
        if len(after) >= 2:
            for i, triple in enumerate(after):
                if triple[1] in coreness_order_model:
                    expected = coreness_order_model[triple[1]]
                    observed = i / (len(after)-1)
                    diff = abs(expected - observed)
                    if diff != 0:
                        coreness_prob -= math.log(diff)
        if not options.p:
            return coreness_prob
    return (pair_prob + coreness_prob) / 2


def generate(AMR, lm=None):
    global seen_vars
    top = AMR.triples(rel=':top')[0]
    sub_root = top
    seen_vars.clear() # for tracking reentrancies

    candidates = generate_subtree(AMR, sub_root, lm)

    # rescore final candidates with bos, eos (i.e. treating them as complete sentences)
    rescored = PriorityQueue()
    while not candidates.empty():
        candidate = candidates.get()
        full_sent_score = -lm.score(candidate[2] + ' .')
        rescored.put((full_sent_score, candidate[0], candidate[2] + ' .'))

    winner = rescored.get()[2]

    # postprocess
    winner = winner[0].upper() + winner[1:] # shouldn't matter for scoring but capitalized looks nicer

    return winner


def generate_subtree(AMR, sub_root, lm):
    global seen_vars

    all_children = AMR.triples(head=sub_root[2])
    children = list(filter(lambda t: t[1] not in {':instance-of', ':wiki', ':mode', ':name'}, all_children))

    if len(children) == 0 or sub_root[2] in seen_vars: # leaf node!
        candidates = var_to_english(AMR, sub_root)

        return candidates

    else:

        # cube pruning
        subtree_q = PriorityQueue()
        cube = []

        # orderings
        order_q = PriorityQueue()
        if not options.p and not options.c:
            root_penult = children[:-1] + [(None, 'ROOT', sub_root[2])] + children[-1:]
            order_q.put((.5, 0, root_penult))
            root_first = [(None, 'ROOT', sub_root[2])] + children
            order_q.put((.5, 1, root_first))
            root_last = children + [(None, 'ROOT', sub_root[2])]
            order_q.put((.5, 2, root_last))
        else:
            if len(children) <= 5:
                for i, order in enumerate(list(permutations(children + [(None, 'ROOT', sub_root[2])]))):
                    score = score_order(order)
                    order_q.put((score, i, order)) # i is just to stop it from trying to compare orders and crashing
            else:
                root_penult = children[:-1] + [(None, 'ROOT', sub_root[2])] + children[-1:]
                root_penult_score = score_order(root_penult)
                order_q.put((root_penult_score, 0, root_penult))
                root_first = [(None, 'ROOT', sub_root[2])] + children
                root_first_score = score_order(root_first)
                order_q.put((root_first_score, 1, root_first))
                root_last = children + [(None, 'ROOT', sub_root[2])]
                root_last_score = score_order(root_last)
                order_q.put((root_last_score, 2, root_last))
        cube.append(sorted(order_q.queue))



        cube.append(sorted(var_to_english(AMR, sub_root).queue))

        for child in children:
            cube.append(sorted(generate_subtree(AMR, child, lm).queue))

        indices = (0,) * len(cube) # initialize to 'corner' of cube
        expanded = set()
        frontier = {}
        frontier[indices] = sum(dimension[0][0] for dimension in cube)
        for i in range(options.k):
            # expand thing at dimensions
            new_hypothesis_elements = []
            for dimension, index in zip(cube, indices):
                new_hypothesis_elements.append(dimension[index])
            new_hypothesis_maps = {} # mapping var to string
            new_hypothesis_order = new_hypothesis_elements[0][2]
            new_hypothesis_maps[sub_root[2]] = new_hypothesis_elements[1][2]
            for sub_hypothesis in new_hypothesis_elements[2:]:
                new_hypothesis_maps[sub_hypothesis[1]] = sub_hypothesis[2]
            new_hypothesis_strings = []
            for triple in new_hypothesis_order:
                s = new_hypothesis_maps[triple[2]]
                if len(s) > 0:
                    new_hypothesis_strings.append(new_hypothesis_maps[triple[2]])
            new_hypothesis_string = ' '.join(new_hypothesis_strings)
            new_hypothesis_score = -lm.score(new_hypothesis_string)
            subtree_q.put((new_hypothesis_score, sub_root[2], new_hypothesis_string))

            expanded.add(indices)
            frontier.pop(indices)
            # add new neighbors to frontier
            for i, index in enumerate(indices):
                # ensure not out of range
                if index+1 < len(cube[i]):
                    new_indices = list(indices)
                    new_indices[i] += 1
                    new_indices = tuple(new_indices)
                    if new_indices not in expanded and new_indices not in frontier:
                        new_frontier_hypothesis_elements = []
                        for dimension, index in zip(cube, new_indices):
                            new_frontier_hypothesis_elements.append(dimension[index])
                        initial_score = sum([sub_hypothesis[0] for sub_hypothesis in new_frontier_hypothesis_elements])
                        frontier[new_indices] = initial_score

            # choose best new frontier item to expand next
            if len(frontier) == 0:
                break
            else:
                indices = min(frontier.items(), key=operator.itemgetter(1))[0]

        return subtree_q


def create_hypotheses(var, prefixes, concepts, suffixes):
    q = PriorityQueue()

    if lm is None:
        return prefixes + concepts + suffixes

    for concept in concepts:
        for prefix in prefixes:
            hyp = prefix + concept
            lm_score = -lm.score(hyp, bos=False, eos=False)
            q.put((lm_score, var, hyp))
        for suffix in suffixes:
            hyp = concept + suffix
            lm_score = -lm.score(hyp, bos=False, eos=False)
            q.put((lm_score, var, hyp))
        if len(prefixes) == 0 and len(suffixes) == 0:
            hyp = concept
            lm_score = -lm.score(hyp, bos=False, eos=False)
            q.put((lm_score, var, hyp))
    return q

def frame_realizations(concept_name):
    realizations = []
    realizations += [concept_name, 'is '+ concept_name, 'are '+ concept_name]
    realizations.append(concept_name+'ment')

    if concept_name.endswith('e'):
        realizations.append(concept_name[:-1]+'ing')
    else:
        realizations.append(concept_name+'ing')

    if concept_name.endswith('y'):
        realizations.append(concept_name[:-1]+'ily')
    else:
        realizations.append(concept_name+'ly')

    if concept_name.endswith('y'):
        realizations.append(concept_name[:-1]+'ies')
    else:
        realizations.append(concept_name+'s')

    if concept_name == 'say':
        past = 'said'
    elif concept_name.endswith('e'):
        past = concept_name+'d'
    elif concept_name.endswith('y'):
        past = concept_name[:-1]+'ied'
    else:
        past = concept_name+'ed'
    realizations += [past, 'is '+past, 'are '+past, 'was '+past, 'were '+past, 'have '+past, 'has '+past, 'had '+past]

    if concept_name.endswith('e'):
        realizations += [concept_name[:-1]+'ion', concept_name[:-1]+'ition', concept_name[:-1]+'ation']
    elif concept_name.endswith('t'):
        realizations += [concept_name[:-1] + 'ion', concept_name[:-1] + 'ition', concept_name[:-1] + 'ation']


    return realizations



def var_to_english(AMR, var_triple):

    global seen_vars
    role = var_triple[1]
    var = var_triple[2]

    prefixes = []
    suffixes = []
    if ':prep-' in role:
        prep = role.split('-')[1:]
        prefixes.append(' '.join(prep) + ' ')
    elif role == ':accompanier':
        prefixes.append('with ')
    elif role == ':destination':
        prefixes.append('to ')
    elif role == ':purpose':
        prefixes.append('to ')
    elif role == ':condition':
        prefixes.append('if ')
    elif role == ':compared-to':
        prefixes.append('than ')
    elif role == ':poss':
        suffixes.append(" 's")
        prefixes.append('of ')
    elif role == ':domain':
        suffixes.append(" is")
    elif role == ':location':
        prefixes += ['in ', 'at ', 'by ']

    if type(var) == str:  # for named entities that have already been made strings
        return create_hypotheses(var, prefixes, [var], suffixes)

    if var.is_constant():
        if var._value == '-':
            concepts = ["not", "do n't", "does n't", "did n't", "is n't", "are n't", "wo n't", "have n't", "has n't", "was n't", "were n't"]
            return create_hypotheses(var, prefixes, concepts, suffixes)

        elif role == ':month':
            month_map = {'1': 'January', '2': 'February', '3': 'March', '4': 'April', '5': 'May', '6': 'June', '7': 'July',
                         '8': 'August', '9': 'September', '10': 'October', '11': 'November', '12': 'December'}
            if var._value in month_map:
                var_value = month_map[var._value]
            else:
                var_value = var._value

        elif role == ':value' and AMR.concept(var_triple[0])._name == 'ordinal-entity':
            ordinal_map = {'1': 'first', '2': 'second', '3': 'third', '4': 'fourth', '5': 'fifth', '6': 'sixth', '7': 'seventh',
                           '8': 'eighth', '9': 'ninth', '10': 'tenth', '11': 'eleventh', '12': 'twelfth'}
            if var._value in ordinal_map:
                var_value = ordinal_map[var._value]
            else:
                if var._value.endswith('1'):
                    var_value = var._value + 'st'
                elif var._value.endswith('2'):
                    var_value = var._value + 'nd'
                elif var._value.endswith('3'):
                    var_value = var._value + 'rd'
                else:
                    var_value = var._value + 'th'
        else:
            var_value = var._value

        return create_hypotheses(var, prefixes, [var_value], suffixes)

    if var in seen_vars:
        return create_hypotheses(var, [], [''], []) # non-first mentions of reentrancies are not realized

    else:
        seen_vars.add(var)
        concept = AMR.concept(var)

        all_children = AMR.triples(head=var)

        # if it has ':name' as a child, it's the special named entity construction
        name_children = list(filter(lambda t: t[1] == ':name', all_children))
        if len(name_children) > 0:
            all_name_triples = AMR.triples(head=name_children[0][2])
            name_triples = list(filter(lambda t: t[1].startswith(':op'), all_name_triples))
            name_strings = []
            for name_triple in name_triples:
                name_string = str(name_triple[2])
                if name_string.startswith('"') and name_string.endswith('"'):
                    name_strings.append(name_string[1:-1])
                else:
                    name_strings.append(name_string)
            name_string = ' '.join(name_strings)

            return create_hypotheses(var, prefixes, [name_string], suffixes)

        # if it's a 'person' and has a child that's :arg0-of either have-org-role-91 or have-rel-role-91
        if concept._name == 'person':
            arg0of_children = list(filter(lambda t: t[1] == ':ARG0-of', all_children))
            if len(arg0of_children) > 0:
                for child in arg0of_children:
                    child_name = AMR.concept(child[2])._name
                    if child_name == 'have-org-role-91' or child_name == 'have-rel-role-91':
                        return create_hypotheses(var, prefixes, [''], suffixes)


        concept_map = {'have-concession-91': ['though'], 'contrast-01': ['but'], 'have-condition-91': ['if'],
                       'amr-unknown': ['why'], 'multi-sentence': ['.'], 'possible-01': ['can'], 'obligate-01': ['must'],
                       'recommend-01': ['should'], 'date-entity': ['on', 'in'], 'cause-01': ['because'],
                       'percentage-entity': ['%', 'percent'], 'include-91': ['of the']}
        if concept._name in concept_map:
            return create_hypotheses(var, prefixes, concept_map[concept._name], suffixes)
        if '-91' in concept._name or '-entity' in concept._name or '-quantity' in concept._name:
            return create_hypotheses(var, prefixes, [''], suffixes)

        if concept.is_frame():
            concept_name = ' '.join(concept._name.split('-')[:-1])
            concepts = frame_realizations(concept_name)
            return create_hypotheses(var, prefixes, concepts, suffixes)

        else: # non-frame concept
            concept_name = ' '.join(concept._name.split('-'))

            # pronouns
            if concept_name == 'i':
                if role == ':poss':
                    return create_hypotheses(var, prefixes, ['my'], suffixes)
                else:
                    return create_hypotheses(var, prefixes, ['i', 'me'], suffixes)
            if concept_name == 'we':
                if role == ':poss':
                    return create_hypotheses(var, prefixes, ['our'], suffixes)
                else:
                    return create_hypotheses(var, prefixes, ['we', 'us'], suffixes)
            if concept_name == 'he':
                if role == ':poss':
                    return create_hypotheses(var, prefixes, ['his'], suffixes)
                else:
                    return create_hypotheses(var, prefixes, ['he', 'him'], suffixes)
            if concept_name == 'she':
                if role == ':poss':
                    return create_hypotheses(var, prefixes, ['her'], suffixes)
                else:
                    return create_hypotheses(var, prefixes, ['she', 'her'], suffixes)
            if concept_name == 'they':
                if role == ':poss':
                    return create_hypotheses(var, prefixes, ['their'], suffixes)
                else:
                    return create_hypotheses(var, prefixes, ['they', 'them'], suffixes)
            if concept_name == 'it':
                if role == ':poss':
                    return create_hypotheses(var, prefixes, ['its'], suffixes)
                else:
                    return create_hypotheses(var, prefixes, ['it'], suffixes)
            if concept_name == 'you':
                if role == ':poss':
                    return create_hypotheses(var, prefixes, ['your'], suffixes)
                else:
                    return create_hypotheses(var, prefixes, ['you'], suffixes)

            concepts = [concept_name, 'the ' + concept_name, 'a ' + concept_name, 'an ' + concept_name,
                        concept_name + 's', 'the ' + concept_name + 's']
            return create_hypotheses(var, prefixes, concepts, suffixes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--stack_size', action="store", type=int, dest="k", default=100, help="stack/queue size to use in pruning")
    parser.add_argument('-p', '--pair_model', action="store", dest="p", default=False, help="path to pair-based ordering model (optional)")
    parser.add_argument('-c', '--coreness_model', action="store", dest="c", default=False, help="path to coreness-based ordering model (optional)")
    parser.add_argument('lm', action="store", help="path to language model")
    parser.add_argument('i', action="store", help="path to input file")
    parser.add_argument('o', action="store", help="path to output file")
    options = parser.parse_args()

    seen_vars = set()

    lm=kenlm.Model(options.lm)

    if options.p:
        with open(options.p, 'rb') as pair_order_file:
            pair_order_model = pickle.load(pair_order_file)
    if options.c:
        with open(options.c, 'rb') as coreness_file:
            coreness_order_model = pickle.load(coreness_file)

    with open(options.i) as input_file:
        input = input_file.read().strip()
        sentences = input.split('\n\n')
        output_lines = []
        for sent in sentences:
            lines = sent.splitlines()
            amr_lines = []
            for line in lines:
                if len(line) > 0 and not line.startswith('#'):
                    amr_lines.append(line)
            if len(amr_lines) > 0:
                this_amr = AMR('\n'.join(amr_lines))
                result = generate(this_amr, lm)
                output_lines.append(result + '\n')

    with open(options.o, 'w') as output_file:
        output_file.writelines(output_lines)

