#%%
import random
from collections import deque

class ErrorSimulator:
    """
    A class to simulate edit distance errors in sentences.
    
    """
    
    def __init__(self, buffer_size=1000):
        self.buffer = []
        self.buffer = deque(maxlen=buffer_size)
        
    def random_word(self):
        """ Generate a random word of length."""
        i = random.randint(0, len(self.buffer)-1)
        return self.buffer[i]
    
    def random_error(self, sentence, error_rate=0.1, error_type=None):
        """ Introduce random edit distance errors into a sentence."""
        words = sentence.split()
        self.buffer.extend(words.copy())
        for i, word in enumerate(words):
            if random.random() >= error_rate:
                continue
            error_type = error_type or random.choice(['insertion', 'deletion', 'substitution', 'repetition'])
            if error_type == 'insertion':
                words[i] += " "+ self.random_word()
                # print(f"Insert {words[i]}")
            elif error_type == 'deletion':
                words[i] = ""
                # print(f"Delete {word}")
            elif error_type == 'substitution':
                words[i] = self.random_word()
                # print(f"Substitute {word} => {words[i]}")
            elif error_type == 'repetition':
                words[i] += " " + word
                # print(f"Repeat {word}")

        return  ' '.join(words)

#%%
if __name__ == "__main__":
    simu = ErrorSimulator()
    for err_type in ['insertion', 'deletion', 'substitution', 'repetition']:
        txt = "This is a sample sentence for testing."
        err_txt = simu.random_error(txt, 1, err_type)
        print(f"Error type: {err_type}")
        print("Original:", txt)
        print("Corrupted:", err_txt)
        print("Buffer:", len(simu.buffer))

# %%
