from tqdm import tqdm
import math


class BaseModel(object):
    def __init__(self, degree, context, thresold=3):
        assert degree > 0, 'degree of model should be positive'
        super(BaseModel, self).__init__()
        self.degree = degree
        self.low_thresold = thresold
        self._clear()
        self.preprocess(context)

    def _clear(self):
        self.ngram_num = [0] * self.degree
        self.prob, self.freq = {}, {}
        self.ngram_list = [[] for i in range(self.degree)]

    def replace_low_freq(self, context):
        nfreq = {}
        for i in context:
            nfreq[i] = nfreq.get(i, 0) + 1
        for idx, word in enumerate(context):
            if word == '<s>' or word == '</s>':
                continue
            if nfreq[word] <= self.low_thresold:
                context[idx] = '<UNK>'

    def preprocess(self, context):
        context = self.process_context(context)
        self.word_count(context)

    def process_context(self, context):
        context = context.split(' ')
        if len(context) == 0:
            context = ['<s>', '</s>']

        else:
            if context[0] != '<s>':
                context.insert(0, '<s>')
            if context[-1] != '</s>':
                context.append('</s>')

        self.replace_low_freq(context)
        if '<UNK>' not in context:
            context.insert(-2, '<UNK>')
        return context

    def word_count(self, context):
        context_len = len(context)

        for i in range(context_len):
            for j in range(self.degree):
                if i + j < context_len:
                    key = tuple(context[i: i + j + 1])
                    self.freq[key] = self.freq.get(key, 0) + 1
                    self.ngram_list[j].append(key)
                else:
                    break

        for i in range(self.degree):
            self.ngram_num[i] = len(self.ngram_list[i])

        for k, v in self.freq.items():
            if len(k) == 1:
                self.prob[k] = v / self.ngram_num[len(k) - 1]
            else:
                self.prob[k] = v / self.freq[k[:-1]]

        self.word_set = set(context)
        # print('<s>' in self.word_set, '</s>' in self.word_set)


class AddkModel(BaseModel):
    def __init__(self, degree, context, k=1):
        super(AddkModel, self).__init__(degree, context)
        self.k = k

    def set_k(self, k):
        assert type(k) == int or type(k) == float, "k should be a num"
        assert 0 < k and k <= 1, "value of k should fall in (0, 1]"
        self.k = k

    def get_p(self, context):
        if isinstance(context, str):
            context = tuple(context.split(' '))
        assert len(context) <= self.degree, 'context has higher order than model'

        if len(context) == 1:
            return self.prob.get(context, 0)

        return (self.freq.get(context, 0) + self.k) / \
            (self.freq.get(context[:-1], 0) + self.k * self.ngram_num[0])

    def get_PPL(self, context):
        context = self.process_context(context)
        context = self.mark_unk(context)
        ans, context_len = 0 ,len(context)
        for i in range(1, min(context_len, self.degree)):
            ans += math.log(self.get_p(tuple(context[: i + 1])))
        for i in range(self.degree, context_len):
            key = tuple(context[i - self.degree + 1: i + 1])
            ans += math.log(self.get_p(key))

        return math.exp(-ans / (context_len - 1))



class InterpolationModel(BaseModel):
    def __init__(self, degree, context):
        super(InterpolationModel, self).__init__(degree, context)
        self.group = {}
        self.group_set = [{0} if i > 0 else set() for i in range(self.degree)]
        for k, v in self.freq.items():
            self.group[k] = v
            self.group_set[len(k) - 1].add(v)

        self.lambdas = [{} for i in range(self.degree - 1)]
        self.trained = False

    def clear(self):
        self._clear()
        self.trained = False

    def mark_unk(self, context):
        for idx, word in enumerate(context):
            if word not in self.word_set:
                context[idx] = '<UNK>'
        return context

    def train(self, context, eps=1e-5, verbose=False):
        context = self.process_context(context)
        context = self.mark_unk(context)

        assert len(set(context) - self.word_set) == 0,\
            "unknown word shown in context"

        for dg in range(self.degree - 1):
            if verbose:
                print('training for degree {}'.format(dg + 2))

            self._train_degree(dg, context, eps=eps, verbose=verbose)

            if verbose:
                print('training for degree {} ends'.format(dg + 2))
        self.trained = True

    def get_p(self, context):
        if isinstance(context, str):
            context = tuple(context.split(' '))
        context = tuple(i if i in self.word_set else '<UNK>' for i in context)

        assert len(context) <= self.degree, "context has higher order than model"
        return self._get_p(context, len(context))

    def get_PPL(self, context):
        assert self.trained, 'model should be trained before prediction'
        context = self.process_context(context)
        context = self.mark_unk(context)

        context_len = len(context)
        ans = 0
        for i in range(1, min(self.degree, context_len)):
            ans += math.log(self._get_p(tuple(context[: i + 1]), i + 1))

        for i in range(self.degree, context_len):
            key = tuple(context[i - self.degree + 1: i + 1])
            ans += math.log(self._get_p(key, self.degree))

        return math.exp(-ans / (context_len - 1))

    def _get_p(self, context, degree):
        if degree == 1:
            return self.prob.get(context, 0)
        pm = self._get_p(context[1:], degree - 1)
        group_num = self.group.get(context[:-1])
        nlambda = self.lambdas[degree - 2].get(group_num, 1)
        return nlambda * pm + (1 - nlambda) * self.prob.get(context, 0)

    def _get_answer(self, context, degree, tlambda):
        answer = tlambda * self._get_p(context[1:], degree - 1)
        answer += (1 - tlambda) * self.prob.get(context, 0)
        return answer

    def _get_answer_group(self, context, degree, lmid, rmid, verbose=False):
        context_len = len(context)
        lans, rans = {}, {}
        iterator = range(degree - 1, context_len)
        if verbose:
            iterator = tqdm(iterator)

        for i in iterator:
            key = tuple(context[i - degree + 1: i + 1])
            group_num = self.group.get(key[:-1], 0)
            lans[group_num] = lans.get(group_num, 0) + \
                math.log(self._get_answer(key, degree, lmid[group_num]))
            rans[group_num] = rans.get(group_num, 0) + \
                math.log(self._get_answer(key, degree, rmid[group_num]))
        return lans, rans

    def _train_degree(self, degree, context, eps=1e-5, verbose=False):
        tlf, trt, lmid, rmid = {}, {}, {}, {}
        for gp in self.group_set[degree]:
            tlf[gp], trt[gp] = 0, 1

        total_epoch = 0
        while True:
            if verbose:
                print('[INFO] {} epoch start'.format(total_epoch))

            maxgap = 0
            for gp in self.group_set[degree]:
                lmid[gp] = tlf[gp] + (trt[gp] - tlf[gp]) / 3
                rmid[gp] = trt[gp] - (trt[gp] - tlf[gp]) / 3

            lans, rans = self._get_answer_group(
                context, degree + 2, lmid, rmid,
                verbose=verbose
            )

            for gp in self.group_set[degree]:
                if lans.get(gp, 0) > rans.get(gp, 0):
                    trt[gp] = rmid[gp]
                else:
                    tlf[gp] = lmid[gp]
                maxgap = max(maxgap, trt[gp] - tlf[gp])

            if verbose:
                print('[INFO] {} epoch end with maxgap of {}'.format(
                    total_epoch, maxgap
                ))
                total_epoch += 1

            if maxgap <= eps:
                break

        for gp in self.group_set[degree]:
            self.lambdas[degree][gp] = (trt[gp] + tlf[gp]) / 2
