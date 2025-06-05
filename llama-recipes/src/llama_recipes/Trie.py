import os
import numpy as np
from tqdm import tqdm
from typing import Optional, Union


class NULL:
    pass


class Node:
    def __init__(self, value=NULL):
        self.value = value
        self.children = dict()

    def __len__(self):
        """Return the number of keys in the subtree rooted at this node."""
        return int(self.value is not NULL) + sum(map(len, self.children.values()))

    def __repr__(self):
        return '(%s, {%s})' % (
            repr(self.value) if self.value is not NULL else "NULL",
            ', '.join('%r: %r' % t for t in self.children.items()))

    def __copy__(self):
        clone = self.__class__(self.value)
        clone_children = clone.children
        for key, child in self.children.items():
            clone_children[key] = child.__copy__()
        return clone

    def __getstate__(self):
        return self.value, self.children

    def __setstate__(self, state):
        self.value, self.children = state



class TrieIndex(object):
    """
    TrieIndex Index.
    """
    def __init__(self, rank:int=0, sep_token_id=None, save_dir:str=".", save_name:Optional[str]=None, **kwargs):
        """
        Args:
            save_dir: the directory to save the trie index
            pad_token_id: will replace the -1 in the codes with ``pad_token_id``
        """
        super().__init__()
        self.rank = rank
        self.save_dir = save_dir
        if save_name is None:
            self.save_path = os.path.join(save_dir, type(self).__name__.lower())
        else:
            self.save_path = os.path.join(save_dir, save_name)
        self._root = Node()
        self.sep_token_id = sep_token_id

    def _find(self, key):
        """
        Find the node corresponding to the key.

        Returns:
            Optional[Node]: a valid node if the key is stored, otherwise None
        """
        node = self._root
        for part in key:
            node = node.children.get(part)
            if node is None:
                break
        return node

    def __len__(self):
        return len(self._root)

    def __iter__(self):
        return self.iterkeys()

    def __contains__(self, key):
        node = self._find(key)
        return node is not None and node.value is not NULL

    def __getitem__(self, key):
        node = self._find(key)
        if node is None or node.value is NULL:
            raise KeyError(f"Invalid key {key}")
        return node.value

    def __setitem__(self, key, value):
        """
        Insert Key in the trie; Append the value into the value list.
        """
        node = self._root
        factory = Node
        for part in key:
            # the setdefault function will update the dict
            next_node = node.children.get(part)
            if next_node is None:
                node = node.children.setdefault(part, factory())
            else:
                node = next_node
        # 注意这里的实现允许一个键关联多个值
        if node.value is NULL:
            node.value = [value]
        else:
            node.value.append(value)

    def __delitem__(self, key):
        # 记录路径上的节点和部分
        nodes_parts = []
        node = self._root
        for part in key:
            nodes_parts.append((node, part))
            node = node.children.get(part)
            if node is None:
                break
        # 删除节点值并清理无用分支
        if node is None or node.value is NULL:
            raise KeyError
        node.value = NULL
        while node.value is NULL and not node.children and nodes_parts:
            node, part = nodes_parts.pop()
            del node.children[part]

    def __repr__(self):
        return '%s({%s})' % (
            self.__class__.__name__,
            ', '.join('%r: %r' % t for t in self.iteritems()))

    #----- extended mapping API methods ----------------------------------------
    # pylint: disable=arguments-differ
    def keys(self, prefix=None):
        """Return a list of this trie's keys.
        :param prefix: If not None, return only the keys prefixed by ``prefix``.
        """
        return list(self.iterkeys(prefix))

    def values(self, prefix=None):
        """Return a list of this trie's values.
        :param prefix: If not None, return only the values associated with keys prefixed by ``prefix``.
        """
        return list(self.itervalues(prefix))

    def items(self, prefix=None):
        """Return a list of this trie's items (``(key,value)`` tuples).
        :param prefix: If not None, return only the items associated with keys prefixed by ``prefix``.
        """
        return list(self.iteritems(prefix))

    def iterkeys(self, prefix=None):
        """Return an iterator over this trie's keys.
        :param prefix: If not None, yield only the keys prefixed by ``prefix``.
        """
        return (key for key, value in self.iteritems(prefix))

    def itervalues(self, prefix=None):
        """Return an iterator over this trie's values.
        :param prefix: If not None, yield only the values associated with keys prefixed by ``prefix``.
        """
        def generator(node, null=NULL):
            if node.value is not null:
                yield node.value
            for child in node.children.values():
                for subresult in generator(child):
                    yield subresult
        if prefix is None:
            root = self._root
        else:
            root = self._find(prefix)
            if root is None:
                root = Node()
        return generator(root)

    def iteritems(self, prefix=None):
        """Return an iterator over this trie's items (``(key,value)`` tuples).
        :param prefix: If not None, yield only the items associated with keys prefixed by ``prefix``.
        """
        parts = []
        append = parts.append

        def generator(node, key_factory=list, parts=parts,
                      append=append, null=NULL):
            if node.value is not null:
                yield (key_factory(parts), node.value)
            for part, child in node.children.items():
                append(part)
                for subresult in generator(child):
                    yield subresult
                del parts[-1]

        root = self._root
        if prefix is not None:
            for part in prefix:
                append(part)
                root = root.children.get(part)
                if root is None:
                    root = Node()
                    break

        return generator(root)

    def get_valid_tokens(self, prefix:Union[list,np.ndarray]) -> list:
        """
        Returns valid key at the next position given the ``prefix``.
        """
        node = self._find(prefix)
        if node is None:
            return []
        keys = list(node.children.keys())
        if keys == []:
            return [128009]
        return keys

    def add(self, sequences:Union[list[list],np.ndarray], ids:Union[list,np.ndarray]=None, verbose:bool=True):
        """
        Add a bulk of sequences into the trie.

        Args:
            sequences: usually the document codes generated by :func:`models.BaseModel.BaseModel.generate_code`
            ids: auxillary ids for each document
            verbose: print information
        """
        # if verbose:
        #     self.logger.info("adding nodes to trie...")

        if verbose:
            bar = tqdm(sequences, ncols=100, leave=False)
        else:
            bar = sequences

        for i, sequence in enumerate(bar):
            if ids is not None:
                id = ids[i]
            else:
                id = i

            sequence = sequence[sequence != -1].copy()
            self[sequence] = id

    def save(self, save_path:Optional[str]=None):
        """
        Save the trie at ``save_path``.

        Args:
            save_path: if ``None``, the ``self.save_path`` will be used
        """
        os.makedirs(self.save_dir, exist_ok=True)
        if save_path is None:
            save_path = self.save_path
        # self.logger.info(f"saving trie at {save_path}")
        # save_pickle(self._root, save_path)

    def load(self, save_path:Optional[str]=None):
        """
        Load the trie from ``save_path``.

        Args:
            save_path: if ``None``, the ``self.save_path`` will be used
        """
        if save_path is None:
            save_path = self.save_path
        # self.logger.info(f"loading trie from {save_path}")
        # self._root = load_pickle(save_path)
    
    def fit(self, text_codes:np.ndarray, load_index:bool=False, save_index:bool=False, verbose:bool=True, **kwargs):
        """
        1. Build TrieIndex from codes;
        2. Save TrieIndex if necessary.

        Args:
            text_codes: the codes of texts
            load_index: if ``True``, load the existing trie
            save_index: if ``True``, force to save the new constructed trie
            rebuild_index: if ``False``, default to ``load_index=True`` and ``save_index=False``; if ``True``, default to ``load_index=False`` and ``save_index=False``
        """
        if load_index and os.path.exists(self.save_path):
            self.load()

        else:
            self.add(text_codes, verbose=verbose)            
            os.makedirs(self.save_dir, exist_ok=True)
            self.save()
            

    def inspect_structure(self, verbose=True) -> np.ndarray:
        """
        check the children number in each layer of the trie
        """
        key_list = [[0]]
        children_num_per_layer = []
        i = 0

        while len(key_list):
            children_num_per_layer.append(0)
            new_key_list = []
            for key in key_list:
                children = self.get_valid_tokens(key)
                if len(children):
                    children_num_per_layer[i] += len(children) / len(key_list)
                    for child in children:
                        new_key_list.append(key + [child])

            key_list = new_key_list.copy()
            i += 1

        children_num_per_layer = np.array(children_num_per_layer)
        # if verbose:
        #     self.logger.info(f"children number per layer: {np.round(children_num_per_layer, 3)}")
        #     self.logger.info(f"all leaves count: {np.round(np.prod(children_num_per_layer[:-1]), 1)}")
        return children_num_per_layer



class WordSetIndex(object):
    def __init__(self, save_dir, sep_token_id=None, pad_token_id=0, eos_token_id=1, rank=0, **kwargs) -> None:
        """
        Construct inverted index. Search with set intersection.

        Args:
            save_dir: folder to save inverted index
            sep_token_id: the token id that separates words
        """
        super().__init__()
        self.rank = rank
        self.sep_token_id = sep_token_id
        self.save_dir = save_dir
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
    

    def fit(self, text_codes:np.ndarray, tokenizer=None, separator=" ", load_index:bool=False, **kwargs):
        """ Build the inverted lists.
        """
        docs_path = os.path.join(self.save_dir, "docs.npy")
        inverse_vocab_path = os.path.join(self.save_dir, "inverse_vocab.npy")
        inverted_lists_path = os.path.join(self.save_dir, "inverted_lists.npy")
        vocab = TrieIndex(save_dir=self.save_dir, save_name="vocab")

        if load_index and os.path.exists(inverted_lists_path):
            self.docs = np.load(docs_path)
            self.inverse_vocab = np.load(inverse_vocab_path)
            self.inverted_lists = np.load(inverted_lists_path, allow_pickle=True)
            vocab.load()
            self.vocab = vocab

        else:
            word_idx = 0
            max_subword_count = 0
            max_word_count = 0
            
            inverse_vocab = []

            docs = []
            inverted_lists = []

            # strip off the leading 0
            for i, text_code in enumerate(tqdm(text_codes, ncols=100)):
                if separator == " ":
                    text = tokenizer.decode(text_code[text_code != -1], skip_special_tokens=True)
                    # currently only support space-separated languages
                    words = text.split(separator)

                    for j, word in enumerate(words):
                        words[j] = tokenizer.encode(word, add_special_tokens=False)
                
                else:
                    # collect subwords of a word (words are separated by sep_token_id in text_codes)
                    word = []
                    words = []
                    for c in text_code:
                        if c == self.eos_token_id:
                            words.append(word.copy())
                            word.clear()
                            break
                        word.append(c)
                        if c == self.sep_token_id:
                            words.append(word.copy())
                            word.clear()
                
                # collect word ids of a document
                doc = []
                for word in words:
                    if len(word) > max_subword_count:
                        max_subword_count = len(word)
                    try:
                        # vocab[word] will call the __geiitem__ of TrieIndex, return node.value
                        word_id = vocab[word][0]
                    except KeyError:
                        word_id = word_idx
                        vocab[word] = word_id
                        # very important to add the copy here to make sure the word is deep copied
                        inverse_vocab.append(word.copy())
                        # create new inverted lists for this word
                        inverted_lists.append([])
                        word_idx += 1
                    # since the trie automatically saves the word id in a list, just slice it out
                    doc.append(word_id)
                    # add to inverted lists
                    inverted_lists[word_id].append(i)
                    word.clear()

                # assert len(doc) == 10
                ## assert len(doc) == 5
                if len(doc) > max_word_count:
                    max_word_count = len(doc)
                docs.append(doc)


            # print(word_idx)
            # with open("/mnt/ssd1/hxli/generative_retrieval/minicpm/test/term_test.json", mode="w") as f:
            #     json.dump(inverse_vocab, f, indent=4)
            inverted_lists = np.array([np.array(x, dtype=np.int32) for x in inverted_lists], dtype=object)
            # pad the subwords so that all of them are equal in length
            inverse_vocab = np.array([x + [-1] * (max_subword_count - len(x)) for x in inverse_vocab], dtype=np.int32)
            # there are by default the same number of words within each document
            docs = np.array([x + [-1] * (max_word_count - len(x)) for x in docs], dtype=np.int32)

            # always save the inverted lists because other processes want to load it
            # self.logger.info(f"saving index at {self.save_dir}...")
            os.makedirs(self.save_dir, exist_ok=True)

            # np.save(docs_path, docs)
            # np.save(inverse_vocab_path, inverse_vocab)
            # np.save(inverted_lists_path, inverted_lists, allow_pickle=True)
            # vocab.save()
                
            # self.docs = np.load(docs_path)
            # self.inverse_vocab = np.load(inverse_vocab_path)
            # self.inverted_lists = np.load(inverted_lists_path, allow_pickle=True)
            # vocab.load()
            # self.vocab = vocab

            self.docs = docs
            self.inverse_vocab = inverse_vocab
            self.inverted_lists = inverted_lists
            self.vocab = vocab

        
        # the number of subwords of each word
        self.subword_num = (self.inverse_vocab != -1).sum(-1)

    def get_valid_tokens(self, prev_text_indices:np.ndarray, prev_words:np.ndarray, prefix:Optional[np.ndarray]):
        """
        Get a list of valid tokens for next step generation.

        prev_text_indices: indices of the texts that contain all previously generated words
        prev_words: previously generated words
        prefix: the current word prefix
        """
        # t1 = time.time()
        #print(prev_text_indices)
        if prev_text_indices is not None:
            # find valid words
            valid_words = self.docs[prev_text_indices].reshape(-1)
            valid_words = valid_words[valid_words != -1]
            # get unique words
            # valid_words = np.unique(valid_words)
            # forbid previously generated words
            # valid_words = np.setdiff1d(valid_words, prev_words, assume_unique=True)

            valid_words = np.setdiff1d(valid_words, prev_words)

            # dup_pos = (valid_words[:, None] == prev_words).any(-1)
            # valid_words = valid_words[~dup_pos]

            # return eos in case there are no more words to be decoded
            if len(valid_words) == 0:
                return [self.eos_token_id]
            
            valid_word_tokens = self.inverse_vocab[valid_words]
        else:
            valid_word_tokens = self.inverse_vocab
        
        # t2 = time.time()
        if prefix is None:
            valid_tokens = valid_word_tokens[:, 0]
        else:
            if prev_text_indices is not None:
                prefix_len = len(prefix)
                # find words with same prefix
                valid_word_idx = (valid_word_tokens[:, :prefix_len] == prefix).all(-1)
                # these tokens are valid
                valid_tokens = valid_word_tokens[valid_word_idx][:, prefix_len]
            else:
                valid_tokens = self.vocab.get_valid_tokens(prefix)

        # t3 = time.time()
        # print(t2-t1, t3-t2)
        return valid_tokens


    def get_valid_tokens_from_doc(self, text_idx, prev_words, prefix):
        """
        Get a list of valid tokens for next step generation. The tokens are from a given text.
        """
        doc = self.docs[text_idx]
        # remove padding words
        valid_words = doc[doc != -1]
        # remove duplicated words
        if prev_words is not None:
            valid_words = np.setdiff1d(valid_words, prev_words, assume_unique=True)
            if len(valid_words) == 0:
                return [self.eos_token_id]

        valid_word_tokens = self.inverse_vocab[valid_words]

        if prefix is None:
            # forbid previously generated words
            # return empty token set in case there are no more valid words
            # get the first token of all valid words
            return valid_word_tokens[:, 0]
        else:
            prefix_len = len(prefix)
            # find words with same prefix
            valid_word_idx = (valid_word_tokens[:, :prefix_len] == prefix).sum(-1) == prefix_len
            valid_tokens = valid_word_tokens[valid_word_idx][:, prefix_len]
            return valid_tokens


    def get_word(self, tokens:np.ndarray):
        """
        Get the word corresponding to the tokens. If the tokens do not represent a valid word, return None instead.
        """
        # NOTE: add [0] because the word idxs are stored in list
        try: 
            word = self.vocab[tokens][0]
        except KeyError:
            word = None
        
        return word


    def get_text(self, word, prev_text_indices=None):
        """
        Get indices of the text that contain the word. If prev_text_indices is provided, the intersection is returned.
        """
        text_indices = self.inverted_lists[word]
        if prev_text_indices is not None:
            text_indices = np.intersect1d(text_indices, prev_text_indices, assume_unique=True)
        return text_indices
    

    def get_text_from_scratch(self, words):
        """
        Get text indices
        """
        # print(words)
        for i, word in enumerate(words):
            if i == 0:
                text_indices = self.get_text(word)
            else:
                # print()
                # print(text_indices)
                # print(i)
                # print(word)
                # print(self.get_text(word))
                # print()
                text_indices = np.intersect1d(text_indices, self.get_text(word))
        return text_indices

    def reconstruct(self, text_idx:int, prev_words:np.ndarray, prefix:Optional[np.ndarray]=None):
        """
        Reconstruct the keyword sequence of text_idx with the prev_words and prefix putting at the head.

        Returns:
            tokens(list): the reconstructed token sequence, with the prev_words placing at the head
            append_idx(int): the index of the start of the reconstructed tokens
        """
        words = self.docs[text_idx]
        words = words[words != -1]
        nonoverlap = (words[:, None] == prev_words).sum(-1) == 0
        tokens_nonoverlap = self.inverse_vocab[words[nonoverlap]]

        if prefix is not None:
            prefix_match = ((tokens_nonoverlap[:, :len(prefix)] == prefix).all(-1)).reshape(-1)
            # slice out the first matching word
            prefix_match_indices = np.argwhere(prefix_match)[:, 0]
            try:
                prefix_match_idx = prefix_match_indices[0]
            except IndexError:
                return None, None
            prefix_match[prefix_match_indices[1:]] = False

            tokens_prefix_match = tokens_nonoverlap[prefix_match_idx]
            tokens_prefix_match = tokens_prefix_match[tokens_prefix_match != -1].tolist()

            tokens_nonoverlap = tokens_nonoverlap[~prefix_match].reshape(-1)
            tokens_nonoverlap = tokens_nonoverlap[tokens_nonoverlap != -1].tolist()

            tokens_overlap = self.inverse_vocab[prev_words].reshape(-1)
            tokens_overlap = tokens_overlap[tokens_overlap != -1].tolist()
            tokens = [self.pad_token_id] + tokens_overlap + tokens_prefix_match + tokens_nonoverlap + [self.eos_token_id]

            return tokens, len(tokens_overlap) + len(prefix) + 1

        else:
            tokens_nonoverlap = tokens_nonoverlap[tokens_nonoverlap != -1].tolist()

            tokens_overlap = self.inverse_vocab[prev_words].reshape(-1)
            tokens_overlap = tokens_overlap[tokens_overlap != -1].tolist()
            tokens = [self.pad_token_id] + tokens_overlap + tokens_nonoverlap + [self.eos_token_id]

            return tokens, len(tokens_overlap) + 1