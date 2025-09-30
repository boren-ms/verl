# %%
import re
import random
from collections import deque
from collections.abc import Sequence
import blobfile as bf
from .utils import to_list


def text_norm(word, tn_prob=1.0):
    """Normalize the text by removing special characters and converting to lowercase."""
    if random.random() <= tn_prob:
        word = " ".join(re.findall(r"[\w'\-]+", word))
        word = word.lower()
    return word


def split_text(text, max_len=1):
    """Split the text into pieces of max_len words."""
    words = text.split()
    if max_len <= 1:
        return words
    pieces = []
    while words:
        n = random.randint(1, max_len)
        pieces.append(" ".join(words[:n]))
        words = words[n:]
    return pieces


def tag_piece(piece, tag="*"):
    """Tag the piece with a specified tag."""
    return f"{tag}{piece}{tag}"


def tag_pieces(pieces, tag="*", specified=None, norm=None):
    """Tag the pieces with a specified tag."""
    if specified is None:
        return [tag_piece(p, tag) for p in pieces]
    norm = norm if norm is not None else lambda x: x
    specified = {norm(p) for p in specified}
    return [tag_piece(p, tag) if norm(p) in specified else p for p in pieces]


# def rand_sample(lst, max_num, new=False):
#     """Randomly sample random num <max_num elements from the list."""
#     if new:
#         n = min(max_num, len(lst))
#         n = random.randint(0, n)
#     else:
#         n = random.randint(0, max_num)
#         n = min(n, len(lst))
#     # print(f"Sampling {n} pieces from {len(lst)}[{max_num}]")
#     return random.sample(lst, n)


def gauss_int(min_val, max_val, coverage=0.95):
    """Randomly sample an integer between min_val and max_val."""
    z_table = {0.68: 1.0, 0.90: 1.645, 0.95: 1.96, 0.975: 2.24, 0.99: 2.576, 0.997: 3.0}
    # Find the closest key in z_table to the requested coverage
    closest_coverage = min(z_table.keys(), key=lambda k: abs(k - coverage))
    z = z_table[closest_coverage]
    mu = (min_val + max_val) / 2
    sigma = (max_val - min_val) / (2 * z)
    n = int(random.gauss(mu, sigma))
    n = min(max(n, min_val), max_val)
    return n


def rand_uniq_sample(lst, n_max, n_min=0):
    """Randomly sample random n elements from the list."""
    # ignore new sampling for now
    lst = list(set(lst))
    n_max = min(n_max, len(lst))
    n_min = min(n_min, n_max)
    return random.sample(lst, gauss_int(n_min, n_max))


def rand_sample(lst, n_max, n_min=0):
    """Randomly sample random n elements from the list."""
    # ignore new sampling for now
    n_max = min(n_max, len(lst))
    n_min = min(n_min, n_max)
    return random.sample(lst, random.randint(n_min, n_max))


def read_words(file_path, N=None, tn_prob=1.0):
    """Read the top N lines from a file."""
    words = []
    with bf.BlobFile(file_path, "r") as f:
        for i, line in enumerate(f):
            if N is not None and i >= N:
                break
            word = line.split()[0]
            words.append(text_norm(word, tn_prob))
    return words


def get_range(args):
    """range function that accepts multiple arguments."""
    if isinstance(args, int):
        return 0, args
    elif isinstance(args, Sequence):
        return args[0], args[-1]
    raise ValueError(f"Invalid range argument: {args}. Must be int or tuple/list.")


def random_sample(lst, n):
    n = int(min(n, len(lst)))
    if n <= 0:
        return []
    return random.sample(lst, n)


class PieceSampler:
    """Sample segments from the text using instance parameters."""

    def __init__(
        self,
        buffer_size=100000,
        bias_prob=1.0,
        context_prob=1.0,
        hit_prob=0.5,
        hit_ratio=None,
        miss_prob=1,
        max_piece_len=1,
        sampling_version=None,
        common_word_file=None,
        common_word_num=None,
        sample_range=10,
        tag="*",
        tag_all=True,
        log_interval=None,
        task_info=0,
    ):
        """Initialize the PieceSampler with configuration parameters.

        Args:
            buffer_size: Maximum size of the buffer for negative pieces
            max_len: Maximum words of text piece
            bias_prob: biasing prompt probability
            max_num: Maximum number of pieces to sample
            hit_prob: Probability of sampling from positive words
            miss_prob: Probability of sampling from negative words
        """
        self.buffer = deque(maxlen=buffer_size)
        self.max_piece_len = max_piece_len
        self.bias_prob = bias_prob
        self.ctx_prob = context_prob
        self.sample_range = get_range(sample_range)
        self.hit_prob = hit_prob
        self.hit_ratio = to_list(hit_ratio, (0, 1))
        self.miss_prob = miss_prob
        self.tag = tag
        self.tag_all = tag_all
        self.task_info = task_info
        if sampling_version == "v0":
            self._sample = self._sample_v0
        elif sampling_version == "v1":
            self._sample = self._sample_v1
        elif sampling_version == "v2":
            self._sample = self._sample_v2
        else:
            self._sample = self._old_sample
        self.log_interval = int(log_interval) if log_interval else None
        self.common_words = set(read_words(common_word_file, common_word_num) if common_word_file else [])
        self.idx = 0

    def filter_commons(self, examples):
        """Filter out common words from the examples."""
        if not self.common_words:
            return examples
        new_examples = []
        for phrase in examples:
            if all(wd in self.common_words for wd in text_norm(phrase).split()):
                continue
            new_examples.append(phrase)
        return new_examples

    def _sample_v0(self, pieces):
        """Sample segments from the positive pieces."""
        if self.miss_prob > 0:  # add negative sampling, if miss_prob > 0
            self.buffer.extend(set(pieces))
        if random.random() > self.bias_prob:
            return []
        num_pieces = gauss_int(*self.sample_range)
        examples = []
        if random.random() <= self.hit_prob:
            examples += rand_uniq_sample(pieces, num_pieces)
        if random.random() <= self.miss_prob:
            n_miss = num_pieces - len(examples)
            examples += rand_uniq_sample(self.buffer, n_miss, n_miss)
        return self.filter_commons(examples)

    def _sample_v1(self, pieces):
        """Sample segments from the positive pieces."""
        pieces = list(self.filter_commons(set(pieces)))  # ensure uniq and filter commons
        if self.miss_prob > 0:  # add negative sampling, if miss_prob > 0
            self.buffer.extend(pieces)
        n_egs = random.randint(*self.sample_range)
        if random.random() > self.bias_prob or not pieces or n_egs <= 0:
            return []

        examples = []
        if random.random() <= self.hit_prob:
            n_hit = random.randint(1, min(n_egs, len(pieces)))
            examples += random_sample(pieces, n_hit)
        n_miss = n_egs - len(examples)
        if random.random() <= self.miss_prob and n_miss > 0:
            examples += random_sample(self.buffer, n_miss)
        return examples

    def _sample_v2(self, pieces):
        """Sample segments from the positive pieces."""
        pieces = list(self.filter_commons(set(pieces)))  # ensure uniq and filter commons
        if self.miss_prob > 0:  # add negative sampling, if miss_prob > 0
            self.buffer.extend(pieces)
        n_egs = random.randint(*self.sample_range)
        if random.random() > self.bias_prob or not pieces or n_egs <= 0:
            return []

        examples = []
        if random.random() <= self.hit_prob:
            n_hit = len(pieces) * random.uniform(self.hit_ratio[0], self.hit_ratio[-1])
            examples += random_sample(pieces, min(n_hit, n_egs))
        n_miss = n_egs - len(examples)
        if random.random() <= self.miss_prob and n_miss > 0:
            examples += random_sample(self.buffer, n_miss)
        return examples

    def _old_sample(self, pieces):
        """Sample segments from the positive pieces."""
        if self.miss_prob > 0:  # add negative sampling, if miss_prob > 0
            self.buffer.extend(pieces)
        if random.random() > self.bias_prob:
            return []
        num_pieces = random.randint(*self.sample_range)
        examples = []
        if random.random() <= self.hit_prob:
            examples += rand_sample(pieces, num_pieces)
        if random.random() <= self.miss_prob:
            n_miss = num_pieces - len(examples)
            examples += rand_sample(self.buffer, n_miss, n_miss)
        return self.filter_commons(examples)

    def sample(self, text):
        """Sample segments from the text using instance parameters."""
        pieces = split_text(text, self.max_piece_len)
        examples = self._sample(pieces)
        examples = list(set(examples))
        random.shuffle(examples)
        # Tag the pieces with shared tags
        shared = set(examples) & set(pieces)
        pieces = tag_pieces(pieces, self.tag, shared)
        if self.tag_all:  # tag the input sample pieces as well
            examples = tag_pieces(examples, self.tag)
            shared = tag_pieces(shared, self.tag)
        self.idx += 1
        context, trans = ", ".join(examples), " ".join(pieces)
        if self.log_interval is not None and self.idx % self.log_interval == 0:
            print(f"[{self.idx}] biasing  list: {context}")
            print(f"[{self.idx}] transcription: {trans}")
        if random.random() >= self.ctx_prob:
            context = ""  # ignore context with some probability
        if context:
            leadings = []
            if self.task_info >= 1:
                leadings.append("<|hit|>" if shared else "<|miss|>")
            if self.task_info >= 2:
                leadings += shared
                leadings.append("<|/hit|>" if shared else "<|/miss|>")
            trans = " ".join(leadings + [trans])
        return context, trans, shared


# %%
if __name__ == "__main__":
    sample_utterances = [
        "Hello, how are you?",
        "This is a test sentence.",
        "I love programming in Python!",
        "The quick brown fox jumps over the lazy dog.",
        "Data science is an interdisciplinary field.",
        "Artificial intelligence is transforming many industries.",
        "Learning to code is a valuable skill in today's job market.",
        "The meeting is scheduled for tomorrow at 9 AM.",
        "She sold seashells by the seashore.",
        "Machine learning models require large amounts of data.",
        "Please remember to submit your assignment by Friday.",
        "The restaurant on Main Street serves delicious pasta.",
        "Climate change is affecting ecosystems worldwide.",
        "Thank you for your help with this project.",
        "The concert was canceled due to bad weather.",
        "Neural networks are inspired by the human brain.",
        "Remember to back up your files regularly.",
        "The new movie received excellent reviews from critics.",
        "Quantum computing has the potential to revolutionize technology.",
        "The library closes at 8 PM on weekdays.",
        "Self-driving cars use various sensors to navigate.",
        "Regular exercise is important for maintaining good health.",
        "The art exhibition features works from local artists.",
        "Natural language processing helps computers understand human language.",
        "Don't forget to water the plants while I'm away.",
    ]

    # Create an instance of PieceSampler
    sampler = PieceSampler(
        buffer_size=100,
        max_piece_len=1,
        bias_prob=1,
        sample_range=(100, 100),
        hit_prob=1,
        miss_prob=0,
        log_interval=2,
        common_word_file="/home/boren/data/librispeech_biasing/words/all_words.count.txt",  # Path to a file with common words
        common_word_num=5000,  # Number of common words to read
    )
    # Sample pieces from the utterances
    for text in sample_utterances:
        examples, trans, shared = sampler.sample(text)
        print(f"Original Trans: {text}")
        print(f"Updated  Trans: {trans}")
        print(f"Sampled examples: {examples}")
        print(f"Shared examples: {shared}")
        print("Buffer:", len(sampler.buffer))
        print("-" * 40)
    # # %%

# %%
