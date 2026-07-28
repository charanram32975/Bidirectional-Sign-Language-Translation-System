class SignToText:
    def __init__(self):
        self.current_text = ""
        self.last_letter = None
        self.stable_count = 0
        self.STABLE_THRESHOLD = 10

        self.label_to_letter = {
            "a-samples": "A",
            "b-samples": "B",
            "c-samples": "C",
            "d-samples": "D",
            "e-samples": "E",
            "f-samples": "F",
            "g-samples": "G",
            "h-samples": "H",
            "i-samples": "I",
            "k-samples": "K",
            "l-samples": "L",
            "m-samples": "M",
            "n-samples": "N",
            "o-samples": "O",
            "p-samples": "P",
            "q-samples": "Q",
            "r-samples": "R",
            "s-samples": "S",
            "t-samples": "T",
            "u-samples": "U",
            "v-samples": "V",
            "w-samples": "W",
            "x-samples": "X",
            "y-samples": "Y"
        }

    def convert(self, label):
        clean = label.strip().lower()
        letter = self.label_to_letter.get(clean)

        if not letter:
            self.last_letter = None
            self.stable_count = 0
            return self.current_text

        if letter == self.last_letter:
            self.stable_count += 1
        else:
            self.last_letter = letter
            self.stable_count = 1

        if self.stable_count == self.STABLE_THRESHOLD:
            self.current_text += letter
            self.last_letter = None
            self.stable_count = 0

        return self.current_text
