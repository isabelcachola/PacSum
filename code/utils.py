from pyrouge import Rouge155
import rouge
import os, shutil, random, string
import json
import os
import numpy as np
import json 
import random
import string
import shutil
import files2rouge
import time
from os.path import join

from gensim_preprocess import preprocess_documents

def save_output(summaries, references, paper_ids, outpath, r, to_stories=False):
    if to_stories:
        for s, r, i in zip(summaries, references, paper_ids):
            with open(os.path.join(outpath, f'{i}.story'), 'w') as f:
                s = list(filter(lambda x:1 if x!='' or x!='\n' else 0, s)) # Remove newlines and empty strings
                story = "\n\n".join(s) + '\n\n'
                summary = f'@highlight\n\n{r[0][0]}'
                f.write(story + summary)
    else:
        with open(outpath + '.hypo', 'w') as fhypo:
            with open(outpath + '.ref', 'w') as fref:
                with open(outpath + '.ids', 'w') as fids:
                    for i, s, r in zip(paper_ids, summaries, references):
                        # f.write(f'Paper ID: {i}\nPrediction: {s}\nTarget: {r}\n\n')
                        fids.write(str(i) + '\n')
                        fhypo.write(str(s).replace('\n', '') + '\n')
                        fref.write(str(r) + '\n')


def test_rouge(cand, ref, tmp_dir='/tmp/', paper_ids=None):
    def random_string(stringLength=8):
        """Generate a random string of fixed length """
        letters= string.ascii_lowercase
        return ''.join(random.sample(letters,stringLength))
    tmp_path = join(tmp_dir, 'tmp'+random_string())
    os.makedirs(tmp_path)
    # print(tmp_path)
    hyp_path = join(tmp_path, 'hyp.txt')
    ref_path = join(tmp_path, 'ref.txt')
    if type(cand) is not list:
        candidates = [line.strip().lower() for line in open(cand, encoding='utf-8')]
    if type(ref) is not list:
        references = [json.loads(line.strip())['target'] for line in open(ref, encoding='utf-8')]
        paper_ids = [json.loads(line.strip())['paper_id'] for line in open(ref, encoding='utf-8')]
    else:
        assert paper_ids is not None
        candidates = cand
        references = ref
    assert len(candidates) == len(references), f'{temp_dir}: len cand {len(candidates)} len ref {len(references)}'

    cnt = len(candidates)
    # evaluator = rouge.Rouge()

    all_scores = []
    save_scores = []

    # For each prediction
    for cand_idx, cand in enumerate(candidates):
        curr_targets = references[cand_idx]
        curr_scores = []
        if type(curr_targets) == str:
            curr_targets = [curr_targets]
        hyp = open(join(tmp_path, 'hyp.txt'), 'w')
        cand = cand.replace('\n', '')
        hyp.write(cand)
        hyp.close()
        # import ipdb; ipdb.set_trace()
        if len(curr_targets) > 0:
            for tgt in curr_targets:
                tgt = tgt.lower().replace('\n', '')
                ref = open(join(tmp_path, 'ref.txt'), 'w')
                ref.write(tgt)
                ref.close()
                try:
                    _r = files2rouge.run(ref_path, hyp_path, to_json=True, ignore_empty=True)
                except Exception as e:
                    print(e)
                    exit(0)
                curr_scores.append(_r)
            # Take the max of curr scores
            import ipdb; ipdb.set_rouge()
            r1 = [r['rouge-1']['f'] for r in curr_scores]
            max_idx = r1.index(max(r1))

            save_scores.append({
                            'paper_id': paper_ids[cand_idx],
                            'all_scores': curr_scores,
                            'max_idx': max_idx,
                            'prediction': cand,
                            'target': curr_targets
                                })
            all_scores.append(curr_scores[max_idx])
    # Average across all scores
    avg_scores = {"rouge-1": {
                    "f": [],
                    "p": [],
                    "r":[]
                    },
                "rouge-2": {
                    "f": [],
                    "p": [],
                    "r": []
                    },
                "rouge-l": {
                    "f": [],
                    "p": [],
                    "r": []
                    }
                }
    for score in all_scores:
        for r_type in score.keys():
            for m_type in score[r_type].keys():
                x = score[r_type][m_type]
                # print(x)
                avg_scores[r_type][m_type].append(x)
    #import ipdb; ipdb.set_trace()          
    for r_type in avg_scores.keys():
        for m_type in avg_scores[r_type].keys():
            x = avg_scores[r_type][m_type]
            avg_scores[r_type][m_type] = np.mean(x)

    shutil.rmtree(tmp_path)
    return avg_scores

def evaluate_rouge(summaries, references, remove_temp=False, rouge_args=[]):
    '''
    Args:
        summaries: [[sentence]]. Each summary is a list of strings (sentences)
        references: [[[sentence]]]. Each reference is a list of candidate summaries.
        remove_temp: bool. Whether to remove the temporary files created during evaluation.
        rouge_args: [string]. A list of arguments to pass to the ROUGE CLI.
    '''
    temp_dir = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
    temp_dir = os.path.join("temp",temp_dir)
    print(temp_dir)
    system_dir = os.path.join(temp_dir, 'system')
    model_dir = os.path.join(temp_dir, 'model')
    # directory for generated summaries
    os.makedirs(system_dir)
    # directory for reference summaries
    os.makedirs(model_dir)
    print(temp_dir, system_dir, model_dir)

    assert len(summaries) == len(references)
    for i, (summary, candidates) in enumerate(zip(summaries, references)):
        summary_fn = '%i.txt' % i
        for j, candidate in enumerate(candidates):
            candidate_fn = '%i.%i.txt' % (i, j)
            with open(os.path.join(model_dir, candidate_fn), 'w') as f:
                #print(candidate)
                f.write('\n'.join(candidate))

        with open(os.path.join(system_dir, summary_fn), 'w') as f:
            f.write('\n'.join(summary))

    args_str = ' '.join(map(str, rouge_args))
    rouge = Rouge155(rouge_args=args_str)
    rouge.system_dir = system_dir
    rouge.model_dir = model_dir
    rouge.system_filename_pattern = '(\d+).txt'
    rouge.model_filename_pattern = '#ID#.\d+.txt'

    #rouge_args = '-c 95 -2 -1 -U -r 1000 -n 4 -w 1.2 -a'
    #output = rouge.convert_and_evaluate(rouge_args=rouge_args)
    output = rouge.convert_and_evaluate()

    r = rouge.output_to_dict(output)
    print(output)
    #print(r)

    # remove the created temporary files
    #if remove_temp:
    #    shutil.rmtree(temp_dir)
    return r

def clean_text_by_sentences(text):
    """Tokenize a given text into sentences, applying filters and lemmatize them.

    Parameters
    ----------
    text : str
        Given text.

    Returns
    -------
    list of :class:`~gensim.summarization.syntactic_unit.SyntacticUnit`
        Sentences of the given text.

    """
    original_sentences = text
    filtered_sentences = [join_words(sentence) for sentence in preprocess_documents(original_sentences)]

    return filtered_sentences


def join_words(words, separator=" "):
    """Concatenates `words` with `separator` between elements.

    Parameters
    ----------
    words : list of str
        Given words.
    separator : str, optional
        The separator between elements.

    Returns
    -------
    str
        String of merged words with separator between elements.

    """
    return separator.join(words)
