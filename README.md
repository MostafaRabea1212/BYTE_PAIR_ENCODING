# BYTE PAIR ENCODING (BPE)


**BPE Tokenizer**

> ملحوظة: كل الشرح الموجود في هذا الملف مأخوذ من التعليقات (الـ Markdown cells) الموجودة فعليًا داخل النوتبوك `BYTE_PAIR_ENCODING_.ipynb`، وتم فقط تجميعها وترتيبها هنا.

---

## جدول المحتويات

1. [Unicode](#unicode)
2. [get_stats](#get_stats)
3. [Tokenizer vs LLM](#tokenizer-vs-llm)
4. [Decoding](#decoding)
5. [BPE: bytes, token ids, merges, vocab(فهم)](#فهم-bpe-bytes-token-ids-merges-vocab)
6. [Encoding](#encoding)
7. [Sentencepiece](#sentencepiece)
8. [encoding="utf-8"](#encodingutf-8)
9. [SentencePiece Training Parameters](#sentencepiece-training-parameters)
10. [Llama 2 tokenizer proto](#llama-2-tokenizer-proto)
11. [vocab_size](#vocab_size)
12. [Final recommendations](#final-recommendations)
13. [Also worth looking at](#also-worth-looking-at)
14. [Vocabulary Size in LLMs (شرح مفصل)](#vocabulary-size-in-llms-شرح-مفصل)

---

## Unicode

هذا الجزء من النوتبوك يبدأ بأمثلة عملية على الفرق بين:
- عدد الـ Unicode code points في نص معين (باستخدام `ord()` و `list()`).
- عدد الـ bytes بعد عمل `encode("UTF-8")` لنفس النص.

مثال النص المستخدم للتجربة (مأخوذ من مصدر خارجي مذكور داخل الكود):

```
# text from https://www.reedbeta.com/blog/programmers-intro-to-unicode/
```

كما تم استخدام نص أطول لاحقًا لأغراض تدريبية بحيث تكون إحصائيات التوكنز أكثر تمثيلًا للواقع:

```
# making the training text longer to have more representative token statistics
# text from https://www.reedbeta.com/blog/programmers-intro-to-unicode/
```

---

## get_stats

```
get_stats
دي بتحسب عدد مرات ظهور كل زوج متجاور.
```

هذه الدالة (`get_stats`) هي المسؤولة عن حساب عدد مرات تكرار كل زوج (pair) من الأرقام المتجاورة في قائمة الـ tokens، وهي الخطوة الأولى في خوارزمية BPE لمعرفة أي زوج هو الأكثر تكرارًا حتى يتم دمجه.

---

## Tokenizer vs LLM

> Note, the Tokenizer is a completely separate, independent module from the LLM. It has its own training dataset of text (which could be different from that of the LLM), on which you train the vocabulary using the Byte Pair Encoding (BPE) algorithm. It then translates back and forth between raw text and sequences of tokens. The LLM later only ever sees the tokens and never directly deals with any text.

### Tokenizer vs LLM

الـ **Tokenizer** والـ **LLM** هما وحدتان منفصلتان تمامًا.

#### Tokenizer

الـ Tokenizer له داتا سيت نصية خاصة به.

يستخدم **BPE (Byte Pair Encoding)** لبناء الـ vocabulary.

بعدها يقوم بمهمتين:

- **Text → Tokens**
- **Tokens → Text**

مثال:

```text
"hello world"
      ↓
Tokenizer
      ↓
[154, 78, 205]
```

---

## Decoding

### decoding

Given a sequence of integers in the range [0, vocab_size], what is the text?

(بمعنى آخر: عملية الـ decoding هي عكس الـ encoding — نأخذ قائمة من الأرقام الصحيحة الموجودة في نطاق `[0, vocab_size]` ونرجّعها إلى النص الأصلي).

---

## فهم BPE: bytes, token ids, merges, vocab

### Understanding BPE: bytes, token ids, merges, and vocab

**Raw bytes**: numbers from 0–255, each representing one original byte (character).
Example: `97` = 'a', `98` = 'b'

**Token id**: a new number assigned after merging two tokens. It's just a counter
(256, 257, 258, ...) — it has **no math relation** to the numbers it came from.

**`merges` dict**: maps a pair of adjacent tokens → the new id that replaces them.
```python
merges = {(97, 98): 256}
```
Meaning: "whenever 97 is followed by 98, replace both with 256."

**`vocab` dict**: maps a token id → the actual bytes it represents.
```python
vocab[256] = vocab[97] + vocab[98]  # b'a' + b'b' = b'ab'
```
This `+` is **concatenation** (joining bytes), not addition. So `256` does NOT
mean `97 + 98` numerically — it's just a new label, while the *real* meaning
lives in `vocab` as the joined byte sequence.

### Example walkthrough
Start: `ids = [97, 98, 97, 98, 99, 97, 98]` (a b a b c a b)

**Step 1** — most frequent pair is `(97, 98)` → new id `256`

**Step 2** — most frequent pair is `(256, 256)` → new id `257`

### Summary table

| Concept | What it really is |
|---|---|
| Raw byte | 0–255, one original character |
| Token id (256+) | just a counter, no math meaning |
| `merges` | pair → new id (which pair becomes which id) |
| `vocab` | id → actual bytes (grows by concatenation, not addition) |

---

## Encoding

### Encodes input text into token IDs using pre-trained merge rules.

داخل دالة الـ `encode`:

```python
pair = min(stats, key=lambda p: merges.get(p, float("inf")))
```

> "It doesn't select the pair with the highest frequency; it selects the pair with the lowest rank/index in merges."

(بمعنى: أثناء عملية الـ encode، الخطوة لا تختار الزوج الأكثر تكرارًا في النص الحالي، وإنما تختار الزوج الذي له أقل رقم/رتبة (rank) داخل قاموس الـ `merges`، لأن هذا يعني أنه كان أول زوج تم دمجه أثناء التدريب، وبالتالي يجب تطبيقه أولًا).

كما يوضح تعليق آخر داخل الكود المكافئ عند قراءة ملفات GPT-2:

```
# ^---- ~equivalent to our "merges"
```

---

## Sentencepiece

### sentencepiece

Commonly used because (unlike tiktoken) it can efficiently both train and inference BPE tokenizers. It is used in both Llama and Mistral series.

[sentencepiece on Github link](https://github.com/google/sentencepiece).

**The big difference**: sentencepiece runs BPE on the Unicode code points directly! It then has an option `character_coverage` for what to do with very very rare codepoints that appear very few times, and it either maps them onto an UNK token, or if `byte_fallback` is turned on, it encodes them with utf-8 and then encodes the raw bytes instead.

TLDR:

- tiktoken encodes to utf-8 and then BPEs bytes
- sentencepiece BPEs the code points and optionally falls back to utf-8 bytes for rare code points (rarity is determined by character_coverage hyperparameter), which then get translated to byte tokens.

(Personally I think the tiktoken way is a lot cleaner...)

---

## encoding="utf-8"

```
encoding="utf-8"
```

دي معناها لما Python يتعامل مع النص الموجود في الملف، استخدم **UTF-8** كطريقة لترميز وفك ترميز النص.

يعني:

```
Python String
     ↓
  UTF-8
     ↓
  Bytes
     ↓
   File
```

وعند القراءة:

```
File
 ↓
Bytes
 ↓
UTF-8
 ↓
Python String
```

---

## SentencePiece Training Parameters

### `normalization_rule_name="identity"`

كلمة **`identity`** تعني: **اترك النص كما هو بالضبط**.

نماذج الذكاء الاصطناعي الحديثة (LLMs) تكره تنظيف النص المسبق وتفضل رؤية النص بحالته الأصلية.

---

### `remove_extra_whitespaces=False`

يمنع المكتبة من حذف المسافات المتكررة، لأن المسافات المتعددة مهمة جداً خصوصاً في الأكواد البرمجية كمسافات الإزاحة **Indentation**.

---

### `input_sentence_size`

الحد الأقصى لعدد الجمل التي سيقرأها من الملف (رقم ضخم لضمان قراءة الملفات الكبيرة).

---

### `max_sentence_length`

أقصى طول للجملة الواحدة بالبايتات **(4192 بايت)**.

---

### `seed_sentencepiece_size`

عدد الرموز الأولية التي يتم الاحتفاظ بها في الذاكرة قبل بدء الدمج والتصفية.

---

### `shuffle_input_sentence=True`

خلط الجمل عشوائياً قبل التدريب لتحسين جودة التوزيع وتجنب التحيز.

---

### `character_coverage`

يطلب من الموديل تغطية **99.995%** من الحروف الموجودة في النص، وتجاهل الحروف النادرة جداً (الشاذة) لتوفير مساحة القاموس.

---

### `byte_fallback=True`

**أهم سطر!**

إذا واجه الموديل حرفاً نادراً جداً لم يتدرب عليه (تم استبعاده)، لا تقم بتحويله إلى `<unk>` المجهول، بل حوله إلى بايتات **(UTF-8 Bytes)** حتى لا يفقد الموديل البيانات.

---

### `split_digits=True`

يفصل الأرقام الفردية.

مثلاً:

`123`

لا تندمج كتوكن واحد، بل كأرقام منفصلة:

`1` و `2` و `3`

مما يجعل الموديل أقوى في العمليات الحسابية.

---

### `split_by_unicode_script`

يمنع دمج حروف من لغات مختلفة.

مثلاً: يمنع دمج حرف عربي مع إنجليزي في توكن واحد.

---

### `split_by_whitespace` & `split_by_number`

يفصل عند المسافات وعند تغير نوع البيانات:

**حروف ثم أرقام.**

---

### `add_dummy_prefix=True`

يضيف **"مسافة وهمية"** في بداية كل نص، حتى يتم اعتبار الكلمة الأولى في الجملة كأنها في المنتصف ومسبوقة بمسافة.

وهذا يمنع تكرار الكلمات في القاموس.

---

### `allow_whitespace_only_pieces`

يسمح للموديل بعمل توكنز مكونة من مسافات بيضاء فقط.

مفيد جداً في كتابة الأكواد.

---

## Special Token IDs

تحديد الأرقام المرجعية **(IDs)** للرموز الخاصة في القاموس:

### `unk_id=0`

رمز المجهول **Unknown** (إجباري في SentencePiece).

---

### `bos_id=1`

رمز بداية الجملة **Begin Of Sentence** (`<s>`).

---

### `eos_id=2`

رمز نهاية الجملة **End Of Sentence** (`</s>`).

---

### `pad_id=-1`

رمز الحشو **Padding**.

ووضع القيمة `-1` يعني إيقاف تفعيله (لا يوجد رمز حشو).

---

## Llama 2 tokenizer proto

If you'd like to export the raw protocol buffer for the `tokenizer.model` released by meta, this is a [helpful issue](https://github.com/google/sentencepiece/issues/121). And this is the result:

```
normalizer_spec {
  name: "identity"
  precompiled_charsmap: ""
  add_dummy_prefix: true
  remove_extra_whitespaces: false
  normalization_rule_tsv: ""
}

trainer_spec {
  input: "/large_experiments/theorem/datasets/MERGED/all.test1.merged"
  model_prefix: "spm_model_32k_200M_charcov099995_allowWSO__v2"
  model_type: BPE
  vocab_size: 32000
  self_test_sample_size: 0
  input_format: "text"
  character_coverage: 0.99995
  input_sentence_size: 200000000
  seed_sentencepiece_size: 1000000
  shrinking_factor: 0.75
  num_threads: 80
  num_sub_iterations: 2
  max_sentence_length: 4192
  shuffle_input_sentence: true
  max_sentencepiece_length: 16
  split_by_unicode_script: true
  split_by_whitespace: true
  split_by_number: true
  treat_whitespace_as_suffix: false
  split_digits: true
  allow_whitespace_only_pieces: true
  vocabulary_output_piece_score: true
  hard_vocab_limit: true
  use_all_vocab: false
  byte_fallback: true
  required_chars: ""
  unk_id: 0
  bos_id: 1
  eos_id: 2
  pad_id: -1
  unk_surface: " \342\201\207 "
  unk_piece: "<unk>"
  bos_piece: "<s>"
  eos_piece: "</s>"
  pad_piece: "<pad>"
  train_extremely_large_corpus: false
  enable_differential_privacy: false
  differential_privacy_noise_level: 0.0
  differential_privacy_clipping_threshold: 0
}
```

---

## vocab_size

- Q: what should be vocab size?
- Q: how can I increase vocab size?
- A: let's see. Reminder: [gpt.py](https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py) from before.

---

## Final recommendations

- Don't brush off tokenization. A lot of footguns and sharp edges here. Security issues. Safety issues.
- Eternal glory to anyone who can delete tokenization as a required step in LLMs.
- In your own application:
  - Maybe you can just re-use the GPT-4 tokens and tiktoken?
  - If you're training a vocab, ok to use BPE with sentencepiece. Careful with the million settings.
  - Switch to minbpe once it is as efficient as sentencepiece :)

---

## Also worth looking at

- [Huggingface Tokenizer](https://huggingface.co/docs/transformers/main_classes/tokenizer). I didn't cover it in detail in the lecture because the algorithm (to my knowledge) is very similar to sentencepiece, but worth potentially evaluating for use in practice.

---

## Vocabulary Size in LLMs (شرح مفصل)

### 1. `vocab_size`

هو عدد الـ **Tokens المختلفة** التي يستطيع الـ Tokenizer إنتاجها.

مثلاً:

`vocab_size = 50,000`

يعني عندنا 50,000 Token مختلفة.

---

### 2. أين يظهر `vocab_size` في الـ Transformer؟

يظهر بشكل أساسي في مكانين:

- **Token Embedding**
- **LM Head**

#### Token Embedding

الـ Embedding Table شكلها:

`vocab_size × embedding_dim`

كل Token له Vector يتم تدريبه.

كلما زاد `vocab_size` → زاد عدد الـ Rows → زادت الـ Parameters.

#### LM Head

الـ LM Head ينتج **Logit لكل Token** يمكن أن يأتي بعد الـ Token الحالي.

لذلك:

`عدد الـ Outputs = vocab_size`

كلما زاد `vocab_size` → زاد عدد الحسابات والـ Parameters.

---

### 3. لماذا لا نجعل `vocab_size` ضخمًا جدًا؟

#### المشكلة الأولى: Parameters و Computation

Vocabulary أكبر يعني:

- Embedding أكبر
- LM Head أكبر
- Parameters أكثر
- Computation أكثر

#### المشكلة الثانية: Undertraining

لو عندنا Vocabulary ضخمة جدًا، بعض الـ Tokens ستظهر نادرًا في الـ Training Data.

`Token نادر → أمثلة تدريب قليلة → Updates قليلة → Undertrained`

#### المشكلة الثالثة: Tokens كبيرة جدًا

Vocabulary أكبر → Tokens أكبر → Sequence أقصر.

هذا جيد لأنه يسمح للموديل بمعالجة نص أكثر.

لكن لو الـ Tokens أصبحت كبيرة جدًا، قد يتم ضغط كمية معلومات كبيرة داخل Token واحد.

`معلومات كثيرة → Token واحد كبير`

وبالتالي قد لا يحصل الـ Transformer على عدد كافٍ من الـ positions لمعالجة التفاصيل.

---

### 4. Vocabulary Size = Trade-off

#### Vocabulary صغيرة

`Tokens أكثر → Sequence أطول → Computation أكثر`

لكن الـ Tokens تكون أصغر وأكثر تفصيلاً.

#### Vocabulary كبيرة

`Tokens أقل → Sequence أقصر → Computation أقل`

لكن:

- Parameters أكثر
- Tokens نادرة أكثر
- احتمال ضغط معلومات كثيرة داخل Token واحد

لذلك `vocab_size` يعتبر **Empirical Hyperparameter** وليس له رقم مثالي ثابت.

---

### 5. ماذا لو أردنا إضافة Tokens لموديل متدرب؟

يمكن إضافة Tokens جديدة مثل:

`<USER>`

`<ASSISTANT>`

`<TOOL>`

نقوم بـ:

1. **Resize للـ Embedding**
   - إضافة Rows جديدة.
   - تهيئة الـ Parameters الجديدة بقيم صغيرة عشوائية.

2. **Resize للـ LM Head**
   - إضافة Outputs جديدة للـ Tokens الجديدة.

3. يمكن عمل:

`Freeze Base Model`

ثم تدريب الـ Parameters الجديدة فقط.

هذه العملية تعتبر **Minor Model Surgery**.

---

### 6. إضافة Tokens ليست فقط للـ Special Tokens

#### Gist Tokens

الفكرة: نضغط **Long Prompt** إلى عدد قليل من الـ **Tokens الجديدة**.

##### الطريقة

`Long Prompt`
↓
`Teacher Model (Frozen)`
↓
`Target Output`

وفي نفس الوقت:

`Gist Tokens + Question`
↓
`Student Model (Frozen)`
↓
`Prediction`

نحسب:

`Loss = Prediction vs Target`

ثم نعمل **Backpropagation للـ Gist Token Embeddings فقط**.

##### ماذا يتم تدريبه؟

- `Base Model` → ❌ Frozen
- `Model Weights` → ❌ لا تتغير
- `Gist Token Embeddings` → ✅ يتم تدريبها

بعد التدريب:

`Long Prompt`
↓
`Gist Tokens`

وبدل إرسال الـ Prompt الطويل كل مرة، نرسل الـ Gist Tokens فقط.

##### الفكرة الأساسية

الـ Gist Token **لا يحتوي النص نفسه**، وإنما الـ Embedding الخاص به يتعلم تمثيل المعلومات الموجودة في الـ Prompt الطويل.

**Gist Tokens = Prompt Compression باستخدام Learned Token Embeddings.**