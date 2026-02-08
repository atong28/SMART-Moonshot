import json
import selfies as sf

class SelfiesVocab:
    def __init__(self, token_to_id: dict):
        self.token_to_id = token_to_id
        self.id_to_token = {i: t for t, i in token_to_id.items()}

        self.pad_id: int = token_to_id["[PAD]"]
        self.bos_id: int = token_to_id["[BOS]"]
        self.eos_id: int = token_to_id["[EOS]"]
        self.unk_id: int = token_to_id["[UNK]"]

    @classmethod
    def from_json(cls, path: str) -> "SelfiesVocab":
        with open(path, "r", encoding="utf-8") as f:
            token_to_id = json.load(f)
        return cls(token_to_id)

    def encode(self, selfies_str: str) -> list[int]:
        toks = list(sf.split_selfies(selfies_str))
        ids = [self.bos_id]

        for t in toks:
            ids.append(self.token_to_id.get(t, self.unk_id))
        ids.append(self.eos_id)

        return ids

    def decode(self, ids: list[int], strip_special=True) -> str:
        toks = []

        for i in ids:
            tok = self.id_to_token.get(i, "[UNK]")
            if strip_special and tok in {"[PAD]", "[BOS]", "[EOS]"}:
                continue
            toks.append(tok)

        return "".join(toks)

    def get_pad_id(self) -> int:
        return self.pad_id
    
    def get_bos_id(self) -> int:
        return self.bos_id
        
    def get_eos_id(self) -> int:
        return self.eos_id  
        
    def get_unk_id(self) -> int:
        return self.unk_id
    
    def __len__(self):
        return len(self.token_to_id)
