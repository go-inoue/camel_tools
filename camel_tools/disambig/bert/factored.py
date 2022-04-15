# -*- coding: utf-8 -*-

# MIT License
#
# Copyright 2018-2022 New York University Abu Dhabi
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import json
from pathlib import Path
import pickle

from camel_tools.data import CATALOGUE
from camel_tools.morphology.database import MorphologyDB
from camel_tools.morphology.analyzer import Analyzer
from camel_tools.disambig.common import Disambiguator, DisambiguatedWord
from camel_tools.disambig.common import ScoredAnalysis
from camel_tools.disambig.score_function import score_analysis_uniform
from camel_tools.disambig.score_function import FEATURE_SET_MAP

from camel_tools.disambig.bert.unfactored import _BERTFeatureTagger


_SCORING_FUNCTION_MAP = {
    'uniform': score_analysis_uniform
}


def _read_json(f_path):
    with open(f_path) as f:
        return json.load(f)


class BERTFactoredDisambiguator(Disambiguator):
    """A disambiguator using an unfactored BERT model. This model is based on
    *Morphosyntactic Tagging with Pre-trained Language Models for Arabic and
    its Dialects* by Inoue, Khalifa, and Habash. Findings of ACL 2022.
    (https://arxiv.org/abs/2110.06852)

    Args:
        model_path (:obj:`str`): The path to the fine-tuned model.
        analyzer (:obj:`~camel_tools.morphology.analyzer.Analyzer`): Analyzer
            to use for providing full morphological analysis of a word.
        features: :obj:`list`, optional): A list of morphological features
            used in the model. Defaults to 14 features.
        top (:obj:`int`, optional): The maximum number of top analyses to
            return. Defaults to 1.
        scorer (:obj:`str`, optional): The scoring function that computes
            matches between the predicted features from the model and the
            output from the analyzer. If `uniform`, the scoring based on the
            uniform weight is used. Defaults to `uniform`.
        tie_breaker (:obj:`str`, optional): The tie breaker used in the feature
            match function. If `tag`, tie breaking based on the unfactored tag
            MLE and factored tag MLE is used. Defaults to `tag`.
        use_gpu (:obj:`bool`, optional): The flag to use a GPU or not.
            Defaults to True.
        batch_size (:obj:`int`, optional): The batch size. Defaults to 32.
        ranking_cache (:obj:`dict`, optional): The cache dictionary of
            pre-computed scored analyses. Defaults to `None`.
    """

    def __init__(self, model_path, analyzer,
                 features=FEATURE_SET_MAP['feats_14'], top=1,
                 scorer='uniform', tie_breaker='tag', use_gpu=True,
                 batch_size=32, ranking_cache=None):
        self.model = {}
        for feat in features:
            model_path_feat = Path(model_path, feat)
            if model_path_feat.exists():
                self.model[feat] = _BERTFeatureTagger(str(model_path_feat))
        self._analyzer = analyzer
        self.features = features
        self._top = max(top, 1)
        self._scorer = _SCORING_FUNCTION_MAP.get(scorer, None)
        self._tie_breaker = tie_breaker
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self._ranking_cache = ranking_cache
        self._mle = _read_json(f'{model_path}/mle_model.json')

    @staticmethod
    def pretrained(model_name=None, top=1, use_gpu=True, batch_size=32,
                   cache_size=10000, pretrained_cache=True):
        """Load a pre-trained model provided with camel_tools.

        Args:
            model_name (:obj:`str`, optional): Name of pre-trained model to
                load. Three models are available: 'msa', 'egy', and 'glf.
                Defaults to `msa`.
            top (:obj:`int`, optional): The maximum number of top analyses to
                return. Defaults to 1.
            use_gpu (:obj:`bool`, optional): The flag to use a GPU or not.
                Defaults to True.
            batch_size (:obj:`int`, optional): The batch size. Defaults to 32.
            cache_size (:obj:`int`, optional): If greater than zero, then
                the analyzer will cache the analyses for the cache_size most
                frequent words, otherwise no analyses will be cached.
                Defaults to 100000.
            pretrained_cache (:obj:`bool`, optional): The flag to use a
                    pretrained cache that stores ranked analyses.
                    Defaults to True.

        Returns:
            :obj:`BERTFactoredDisambiguator`: Instance with loaded pre-trained
            model.
        """

        model_info = CATALOGUE.get_dataset('DisambigBertFactored', model_name)
        model_config = _read_json(Path(model_info.path, 'default_config.json'))
        model_path = str(model_info.path)
        features = FEATURE_SET_MAP[model_config['feature']]
        db = MorphologyDB.builtin_db(model_config['db_name'], 'a')
        analyzer = Analyzer(db, backoff=model_config['backoff'],
                            cache_size=cache_size)
        scorer = model_config['scorer']
        tie_breaker = model_config['tie_breaker']
        if pretrained_cache:
            cache_info = CATALOGUE.get_dataset('DisambigRankingCache',
                                               model_config['ranking_cache'])
            cache_path = Path(cache_info.path, 'default_cache.pickle')
            with open(cache_path, 'rb') as f:
                ranking_cache = pickle.load(f)
        else:
            ranking_cache = {}

        return BERTFactoredDisambiguator(model_path,
                                         analyzer,
                                         top=top,
                                         features=features,
                                         scorer=scorer,
                                         tie_breaker=tie_breaker,
                                         use_gpu=use_gpu,
                                         batch_size=batch_size,
                                         ranking_cache=ranking_cache)

    @staticmethod
    def pretrained_from_config(config, top=1, use_gpu=True, batch_size=32,
                               cache_size=10000, pretrained_cache=True):
        """Load a pre-trained model with custom config file.

        Args:
            config (:obj:`str`): Config file that defines the model
                details. Defaults to `None`.
            top (:obj:`int`, optional): The maximum number of top analyses
                to return. Defaults to 1.
            use_gpu (:obj:`bool`, optional): The flag to use a GPU or not.
                Defaults to True.
            batch_size (:obj:`int`, optional): The batch size.
                Defaults to 32.
            cache_size (:obj:`int`, optional): If greater than zero, then
                the analyzer will cache the analyses for the cache_size
                most frequent words, otherwise no analyses will be cached.
                Defaults to 100000.
            pretrained_cache (:obj:`bool`, optional): The flag to use a
                pretrained cache that stores ranked analyses.
                Defaults to True.

        Returns:
            :obj:`BERTFactoredDisambiguator`: Instance with loaded
            pre-trained model.
        """

        model_config = _read_json(config)
        model_path = model_config['model_path']
        features = FEATURE_SET_MAP[model_config['feature']]
        db = MorphologyDB(model_config['db_path'], 'a')
        analyzer = Analyzer(db,
                            backoff=model_config['backoff'],
                            cache_size=cache_size)
        scorer = model_config['scorer']
        tie_breaker = model_config['tie_breaker']

        if pretrained_cache:
            cache_path = model_config['ranking_cache']
            with open(cache_path, 'rb') as f:
                ranking_cache = pickle.load(f)
        else:
            ranking_cache = {}

        return BERTFactoredDisambiguator(model_path,
                                         analyzer,
                                         top=top,
                                         features=features,
                                         scorer=scorer,
                                         tie_breaker=tie_breaker,
                                         use_gpu=use_gpu,
                                         batch_size=batch_size,
                                         ranking_cache=ranking_cache)

    def _predict_sentences(self, sentences):
        """Predict the morphosyntactic labels of multiple sentences.

        Args:
            sentences (:obj:`list` of :obj:`list` of :obj:`str`): The input
                sentences.

        Returns:
            :obj:`list` of :obj:`list` of :obj:`dict`: The predicted
            morphosyntactic labels for the given sentences.
        """

        parsed_predictions_dict = {
            feat: self.model[feat].predict(sentences,
                                           batch_size=self.batch_size)
            for feat in self.model.keys()
        }

        # place holder for models without analyzer
        parsed_predictions_dict['lex'] = sentences
        parsed_predictions_dict['diac'] = sentences

        parsed_predictions = []
        for parsed in zip(*parsed_predictions_dict.values()):
            predictions = [
                dict(zip(parsed_predictions_dict, t))
                for t in zip(*parsed)
            ]
            parsed_predictions.append(predictions)

        return parsed_predictions

    def _predict_sentence(self, sentence):
        """Predict the morphosyntactic labels of a single sentence.

        Args:
            sentence (:obj:`list` of :obj:`str`): The input sentence.

        Returns:
            :obj:`list` of :obj:`dict`: The predicted morphosyntactic
            labels for the given sentence.
        """

        parsed_predictions_dict = {
            feat: self.model[feat].predict(
                [sentence], batch_size=self.batch_size)[0]
            for feat in self.model.keys()
        }
        # dict of list to list of dict
        parsed_predictions = [dict(zip(parsed_predictions_dict, t))
                        for t in zip(*parsed_predictions_dict.values())]
        # place holder for models without analyzer
        for word, d in zip(sentence, parsed_predictions):
            d['lex'] = word # copy the word when analyzer is not used
            d['diac'] = word # copy the word when analyzer is not used

        return parsed_predictions

    def _scored_analyses(self, word_dd, prediction):
        bert_analysis = prediction
        analyses = self._analyzer.analyze(word_dd)

        if len(analyses) == 0:
            # if the word is not found in the analyzer,
            # return the predictions from BERT
            return [ScoredAnalysis(0, bert_analysis)]

        scored = [(self._scorer(a, bert_analysis, self._mle,
                                tie_breaker=self._tie_breaker,
                                features=self.features), a)
                  for a in analyses]
        scored.sort(key=lambda s: (-s[0], s[1]['diac']))

        max_score = max(s[0] for s in scored)

        if max_score != 0:
            scored_analyses = [ScoredAnalysis(s[0] / max_score, s[1])
                               for s in scored]
        else:
            # if the max score is 0, do not divide
            scored_analyses = [ScoredAnalysis(s[0], s[1])
                               for s in scored]

        return scored_analyses[:self._top]

    def _disambiguate_word(self, word, pred):
        key = (word, tuple(pred[feat] for feat in self.features))
        if key in self.ranking_cache:
            scored_analyses = self.ranking_cache[key]
        else:
            scored_analyses = self._scored_analyses(word, pred)
            self.ranking_cache[key] = scored_analyses

        return DisambiguatedWord(word, scored_analyses)

    def disambiguate_word(self, sentence, word_ndx):
        """Disambiguates a single word in a sentence.

        Args:
            sentence (:obj:`list` of :obj:`str`): The list of space and
                punctuation seperated list of tokens comprising a given
                sentence.
            word_ndx (:obj:`int`): The index of the word token in `sentence` to
                disambiguate.

        Returns:
            :obj:`~camel_tools.disambig.common.DisambiguatedWord`: The
            disambiguation of the word token in `sentence` at `word_ndx`.
        """

        return self.disambiguate(sentence)[word_ndx]

    def disambiguate(self, sentence):
        """Disambiguate all words in a given sentence.

        Args:
            sentence (:obj:`list` of :obj:`str`): The input sentence.

        Returns:
            :obj:`list` of :obj:`~camel_tools.disambig.common.DisambiguatedWord`: The
            list of disambiguations for each word in the given sentence.
        """

        predictions = self._predict_sentence(sentence)

        return [self._disambiguate_word(w, p)
                for (w, p) in zip(sentence, predictions)]

    def disambiguate_sentences(self, sentences):
        """Disambiguate all words in a list of sentences.

        Args:
            sentences (:obj:`list` of :obj:`list` of :obj:`str`): The input
                sentences.

        Returns:
            :obj:`list` of :obj:`list` of :obj:`~camel_tools.disambig.common.DisambiguatedWord`: The
            list of disambiguations for each word in the given sentence.
        """
        predictions = self._predict_sentences(sentences)
        disambiguated_sentences = []
        for sentence, prediction in zip(sentences, predictions):
            disambiguated_sentence = [
                self._disambiguate_word(w, p)
                for (w, p) in zip(sentence, prediction)
            ]
            disambiguated_sentences.append(disambiguated_sentence)

        return disambiguated_sentences

    def tag_sentences(self, sentences, use_analyzer=True):
        """Predict the morphosyntactic labels of a list of sentences. 

        Args:
            sentences (:obj:`list` of :obj:`list` of :obj:`str`): The input
                sentences.
            use_analyzer (:obj:`bool`): The flag to use an analyzer or not.
                If set to False, we return the original input as diac and lex.
                Defaults to True.

        Returns:
            :obj:`list` of :obj:`list` of :obj:`dict`: The predicted The list
            of feature tags for each word in the given sentences
        """

        if use_analyzer:
            tagged_sentences = []
            for prediction in self.disambiguate_sentences(sentences):
                tagged_sentence = [a.analyses[0].analysis for a in prediction]
                tagged_sentences.append(tagged_sentence)

            return tagged_sentences

        return self._predict_sentences(sentences)

    def tag_sentence(self, sentence, use_analyzer=True):
        """Predict the morphosyntactic labels of a single sentence. 

        Args:
            sentence (:obj:`list` of :obj:`str`): The list of space and
                punctuation seperated list of tokens comprising a given
                sentence.
            use_analyzer (:obj:`bool`): The flag to use an analyzer or not.
                If set to False, we return the original input as diac and lex.
                Defaults to True.

        Returns:
            :obj:`list` of :obj:`dict`: The list of feature tags for each word
            in the given sentence
        """

        if use_analyzer:
            return [a.analyses[0].analysis
                    for a in self.disambiguate(sentence)]

        return self._predict_sentence(sentence)

    def all_feats(self):
        """Return a set of all features produced by this disambiguator.

        Returns:
            :obj:`frozenset` of :obj:`str`: The set all features produced by
            this disambiguator.
        """

        return self._analyzer.all_feats()

    def tok_feats(self):
        """Return a set of tokenization features produced by this
        disambiguator.

        Returns:
            :obj:`frozenset` of :obj:`str`: The set tokenization features
            produced by this disambiguator.
        """

        return self._analyzer.tok_feats()