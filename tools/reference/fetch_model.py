"""Download the pinned model files used by the reference scripts and tests.

Files go to tools/reference/.cache/<repo>/<revision>/ (gitignored). Each file is
checked against the size and SHA-256 recorded in src/catalogue/models.ts, so the
reference and the app always use byte-identical artifacts.

Usage: tools/reference/.venv/bin/python -I tools/reference/fetch_model.py [model ...] [--with-q4]
       (--with-q4 also fetches each model's optional files: the other precision)
       (no model names = all models)
"""

import hashlib
import sys
import urllib.request
from pathlib import Path

# Keep in sync with src/catalogue/models.ts.
MODELS = {
    "smollm2-360m-instruct": {
        "repo": "HuggingFaceTB/SmolLM2-360M-Instruct",
        "revision": "a10cc1512eabd3dde888204e902eca88bddb4951",
        "files": {
            "tokenizer.json": (2104556, "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c"),
            "tokenizer_config.json": (3764, "4ec77d44f62efeb38d7e044a1db318f6a939438425312dfa333b8382dbad98df"),
            "onnx/model_q4f16.onnx": (272737275, "cc63370efc2aca6d5307518b85162777132cc5b8d68eeb8154ea9b5fce09ad46"),
        },
        "optional": {
            "onnx/model_q4.onnx": (387943246, "a4ffda96e65beafc6f6cef0cbcf9fdc1cbdd79c230906bf3897d190547c7a596"),
        },
    },
    "qwen2.5-0.5b-instruct": {
        # The -ONNX-MHA export: fused attention (MultiHeadAttention, RotaryEmbedding).
        # The older onnx-community/Qwen2.5-0.5B-Instruct export does attention with
        # plain MatMuls, which ONNX Runtime's WebGPU kernel gets wrong on Adreno GPUs
        # for some shapes, and its q4f16 file overflows fp16.
        "repo": "onnx-community/Qwen2.5-0.5B-Instruct-ONNX-MHA",
        "revision": "b1e71b59c0c33ee49eb562303578961795624293",
        "files": {
            "tokenizer.json": (9116490, "bb5953180a154511cf4d0fb1fc68a3433f1458aa6db3851ac2609cf41e7d9b5b"),
            "tokenizer_config.json": (7849, "6a14243f0ae04884e597b3f032c05e15de9f417318fdcac24ab4e2680e691430"),
            "onnx/model_q4f16.onnx": (478355493, "2a8e924345337b83796bd98bc130ab0ec0163188d50be55a1bf14b95808c55b8"),
        },
        "optional": {
            "onnx/model_q4.onnx": (777326354, "60e105dee084373137dd810d0d5f47c3a0cfe14c2dd3c25f723fba3169e011f6"),
        },
    },
    "qwen3-0.6b": {
        "repo": "onnx-community/Qwen3-0.6B-ONNX",
        "revision": "1e0a4a196ecabdf9a879664110574563d3f372d3",
        "files": {
            "tokenizer.json": (9117040, "e7a95fce95bf5b0946d0ddb3f9d7caa030b7e850bbe92b0edb26bcf563e9f3d5"),
            "tokenizer_config.json": (10360, "982fa8771b0e9e5ebf7e943631c6fc4af301aa81b509e1ca517a7389e4a51d42"),
            "onnx/model_q4f16.onnx": (569789750, "9e33a5911974174761d0dfdcc0bec975d9c45af0eae5e9eb647b8ba9442a8f91"),
        },
        "optional": {
            "onnx/model_q4.onnx": (919096585, "d43d836fc5e240df9013733ccd214972c5d21bd9ec47e574e4f1e359cf90aed0"),
        },
    },
    "smollm2-1.7b-instruct": {
        "repo": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "revision": "31b70e2e869a7173562077fd711b654946d38674",
        "files": {
            "tokenizer.json": (2104556, "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c"),
            "tokenizer_config.json": (3764, "4ec77d44f62efeb38d7e044a1db318f6a939438425312dfa333b8382dbad98df"),
            "onnx/model_q4f16.onnx": (1108730338, "d94946187fb5f27579f3db4ba21fb7f7466c7cbd18956bd420d3981f75282f9c"),
        },
        "optional": {
            "onnx/model_q4.onnx": (1411969607, "467b7b8f62d99f184d3628d24b8d65c151e331695f6e9ea997616c4e279e9a51"),
        },
    },
    # Added 2026-10-10. Weights over 2 GB sit in onnx/*.onnx_data* files next to the graph;
    # Qwen3.5 has a separate token-embedding graph (embed_tokens_*). Qwen2.5 1.5B and Qwen3 1.7B
    # run their q4 files (their q4f16 files overflow fp16 on GPUs), so those are fetched by default.
    "qwen2.5-1.5b-instruct": {
        "repo": "onnx-community/Qwen2.5-1.5B-Instruct",
        "revision": "d54957e3af1be04546436e874a60fe386041b244",
        "files": {
            "tokenizer.json": (7031673, "a8506e7111b80c6d8635951a02eab0f4e1a8e4e5772da83846579e97b16f61bf"),
            "tokenizer_config.json": (7849, "6a14243f0ae04884e597b3f032c05e15de9f417318fdcac24ab4e2680e691430"),

    
            "onnx/model_q4.onnx": (1787566590, "70c24509f760fed9a8f50391165c46a87660197028a42c4547763f1c173ddea6"),
    },
        "optional": {
            "onnx/model_q4f16.onnx": (1221878940, "19dec9f63488016185ba997d5e4492b5ac5b4f7ef1abb45243a91de958838dcd"),
        },
    },
    "qwen2.5-coder-1.5b-instruct": {
        "repo": "onnx-community/Qwen2.5-Coder-1.5B-Instruct",
        "revision": "774cc908b6c2be23bc4e8d4e401e4f8b8fba7daf",
        "files": {
            "onnx/model_q4f16.onnx": (1344237070, "0e33cbfde4be371039daad0032fb3782170442e6901a25d4deabe96117b89882"),
            "tokenizer.json": (11421896, "9c5ae00e602b8860cbd784ba82a8aa14e8feecec692e7076590d014d7b7fdafa"),
            "tokenizer_config.json": (7849, "6a14243f0ae04884e597b3f032c05e15de9f417318fdcac24ab4e2680e691430"),
        },
        "optional": {
            "onnx/model_q4.onnx": (1915754790, "3be13b6b30c1c152bd36e5a886a36800d404f56494c74e2fdf1d634d175372e1"),
        },
    },
    "qwen3-1.7b": {
        "repo": "onnx-community/Qwen3-1.7B-ONNX",
        "revision": "b85539623180ae64303f37373d942cb0595ca44d",
        "files": {
            "chat_template.jinja": (4116, "87a2728cb8dc9fe424d624542f6060ec05a1d285ebbec578bb078900e33396b5"),
            "tokenizer.json": (9117040, "e7a95fce95bf5b0946d0ddb3f9d7caa030b7e850bbe92b0edb26bcf563e9f3d5"),
            "tokenizer_config.json": (10360, "982fa8771b0e9e5ebf7e943631c6fc4af301aa81b509e1ca517a7389e4a51d42"),

    
            "onnx/model_q4.onnx": (2147212861, "3763020b62164be3d9c31dc6b15aa042052f5a9da846e991f6c45e0f1a6dced7"),
    },
        "optional": {
            "onnx/model_q4f16.onnx": (1426069098, "fb85b44defdf43ace50c7a4937127e889b9f8a0483eb87a6c4151a815c5da3b5"),
        },
    },
    "qwen3-4b": {
        "repo": "onnx-community/Qwen3-4B-ONNX",
        "revision": "5cbfbcacd84ed9e54cea017a4cbddb67cc45337f",
        "files": {
            "chat_template.jinja": (4168, "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8"),
            "onnx/model_q4f16.onnx": (59762833, "d3e946f9e38577411b0251051c91a1f20c3c0c831e1cf76a13c58ce279d950de"),
            "onnx/model_q4f16.onnx_data": (2096005120, "050398248de4fce7b31b2d2caa909596e4c7aa0f35696270df46b8aaf2209fc8"),
            "onnx/model_q4f16.onnx_data_1": (677150720, "363ff5e70ebeb5866afea8eb80b7bdfe22d94d735a8c4d2ebf90c75424c2a410"),
            "tokenizer.json": (9117040, "e7a95fce95bf5b0946d0ddb3f9d7caa030b7e850bbe92b0edb26bcf563e9f3d5"),
            "tokenizer_config.json": (10416, "355151ccb46b01e22d93a1e4c02f675caa99ad08c2e15b770975b7a1fc057fd6"),
        },
        "optional": {
        },
    },
    "llama-3.2-1b-instruct": {
        "repo": "onnx-community/Llama-3.2-1B-Instruct-ONNX",
        "revision": "14007543b6dc92de88daf96a9aa85d2f95ace6ef",
        "files": {
            "chat_template.jinja": (3827, "5816fce10444e03c2e9ee1ef8a4a1ea61ae7e69e438613f3b17b69d0426223a4"),
            "onnx/model_q4f16.onnx": (149790, "fccc573530dbefac92a36797262e4228dc6fb474734d80f6781e37744550333c"),
            "onnx/model_q4f16.onnx_data": (1089605632, "3f11fd184cd0b34f359cc2696c898026dfc0e169e56c3a2505ae964948a873f1"),
            "tokenizer.json": (11574638, "3a223ade375cc1d13b04e897ce1d36a04f50140e1ba3d107021ea68d4b5e614c"),
            "tokenizer_config.json": (54557, "fb8e113b6240ab997fe87464b8b58697cc769a8df40699356e8524d0dfc60c0e"),
        },
        "optional": {
            "onnx/model_q4.onnx": (149112, "a3a6f10916f79379d15cfa9270b7be0d09be2b80fe0872bd7030eaf9001baf46"),
            "onnx/model_q4.onnx_data": (1692672000, "07cc629ef2cb7fdb18615ce2e4f3774f763e6fc840207d772a8b511eead36647"),
        },
    },
    "llama-3.2-3b-instruct": {
        "repo": "onnx-community/Llama-3.2-3B-Instruct-ONNX",
        "revision": "cab364e7d0e1de7aa09e3abc932be92361c5b55f",
        "files": {
            "chat_template.jinja": (3827, "5816fce10444e03c2e9ee1ef8a4a1ea61ae7e69e438613f3b17b69d0426223a4"),
            "onnx/model_q4f16.onnx": (260899, "43648be8ff45ed7bc75c75ea0d495beffa8a8632910e53d3aa824d6bfffaae46"),
            "onnx/model_q4f16.onnx_data": (2095929344, "0669c8c258ea5437b82cc17e5ca87bb91a9ede5b2f5ff80675c0b8e51f1b6043"),
            "onnx/model_q4f16.onnx_data_1": (311427072, "63b1b82298ad66f940b4f918f81c386fbe4e15a4e178efb14bc558d127185113"),
            "tokenizer.json": (11574638, "3a223ade375cc1d13b04e897ce1d36a04f50140e1ba3d107021ea68d4b5e614c"),
            "tokenizer_config.json": (54557, "fb8e113b6240ab997fe87464b8b58697cc769a8df40699356e8524d0dfc60c0e"),
        },
        "optional": {
            "onnx/model_q4.onnx": (259880, "06366bfcd5c1cc2580b2a602b3628b512d3c7404987508e3b4fb7550e820f0ba"),
            "onnx/model_q4.onnx_data": (2094051328, "1e8bde953a04cc6fd5d6a046205b97b6d69e556d34675592f87d17f176ea5245"),
            "onnx/model_q4.onnx_data_1": (1311375360, "0d8666d2aa4678fd8b6e427b11147792368bbac1bbcc28ecc454203e00d00509"),
        },
    },
    "phi-4-mini-instruct": {
        "repo": "onnx-community/Phi-4-mini-instruct-ONNX",
        "revision": "e61f45fc5fabba2aee31ff85ba4cf99219b4bf28",
        "files": {
            "chat_template.jinja": (423, "febf589225c9728ab791f52e8897d7607a823d45368f0a4c92fa68997b40cce9"),
            "onnx/model_q4f16.onnx": (26270832, "ca26127777adf1df99b5fc1a3b4d1e0c426a6bf56626889873bfc4a6a095b4fc"),
            "onnx/model_q4f16.onnx_data": (2087043072, "385526d648e4b3e361f3117564a6bd3cad7712a5c06fa23401763187674fd46c"),
            "onnx/model_q4f16.onnx_data_1": (438239232, "b9a5d6f40fde30e9155d671dc630d2ea554f903f01c4ef8bbdc64c5b38035923"),
            "tokenizer.json": (13303196, "9ca5aa723a31a7a122497e059bd48dd67a5bd03ad16b3ffcf16093fd3021c1eb"),
            "tokenizer_config.json": (766, "e263ca0b737a5e1ffc6bcb8ca1b0c85ae7febc90ecbf1aac968170f0f79b4feb"),
        },
        "optional": {
            "onnx/model_q4.onnx": (52220822, "6bf600a41ee8503ee1a7ed572ba72888067eeaee6bd2200ae42d484fd86eedcb"),
            "onnx/model_q4.onnx_data": (2087319552, "205a7f6a8a254ab8bf1bc1698fe740ec797e2ec64e5a792c650fe717b7554968"),
            "onnx/model_q4.onnx_data_1": (678100992, "4e83ac7f6aa03d7d92c1785e92c3c62e9419ec5d740a7f411608d48ecdf00ecf"),
        },
    },
    "phi-3.5-mini-instruct": {
        "repo": "onnx-community/Phi-3.5-mini-instruct-onnx-web",
        "revision": "1a2c0e4b2eb1a872716fdc50b100617499739a73",
        "files": {
            "onnx/model_q4f16.onnx": (220452346, "1e1119b8d3fa1adaaad4abaf0c0405dec377054948440e02dd6837d18d2a920d"),
            "onnx/model_q4f16.onnx_data": (2097020928, "b6dfb3069b717b518266dd58cb8e794bc0712fbc46e1737aef99cd8875302cf0"),
            "tokenizer.json": (1844408, "9e2ae3d66819f163cdcfedba5078f3a3af8118c2712491ed60912f77886bda6f"),
            "tokenizer_config.json": (3984, "92a6e3f894075fda19f47d8755c6ce65b2a974716307d13ae5d79196011ebb44"),
        },
        "optional": {
        },
    },
    "qwen3.5-0.8b": {
        "repo": "onnx-community/Qwen3.5-0.8B-ONNX",
        "revision": "d1ea95774d40a29f0479bf9138cdf488a0394b6b",
        "files": {
            "chat_template.jinja": (7755, "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80"),
            "onnx/decoder_model_merged_q4f16.onnx": (1036898, "34e17c8e2035919df86ab1f52b41999a1bd18ba96b49dba6ac8d340aae652006"),
            "onnx/decoder_model_merged_q4f16.onnx_data": (436662272, "468cf83a51e81e27ffb4210268b1b09979e68dd128ad5fe347e5d08721cecc41"),
            "onnx/embed_tokens_q4f16.onnx": (1064, "8218531ac44ae9978d50647f1d907c53c308f758514b992504238c77843c254d"),
            "onnx/embed_tokens_q4f16.onnx_data": (147005440, "ec4a1f13ff942653b52000a7a0ec40504110d8be9a0ecab2da4d3063588ed563"),
            "tokenizer.json": (19226111, "89da80cc6689bef4d90cc1028249436975ffb0814618f1d93c65310e05801a9b"),
            "tokenizer_config.json": (10060, "c4cd279f8fed1ddea0d23ab13e7ec255b15fcad0200ec0487fb853d880b0b40b"),
        },
        "optional": {
            "onnx/decoder_model_merged_q4.onnx": (876359, "cfb338c5112e11e2a123d6d03d8870cd800883ee333ec0a25289482b609a16f8"),
            "onnx/decoder_model_merged_q4.onnx_data": (485425152, "fb38200610a2206aa4c10f4c5f41c08107e2b0b44968166943b3937a1eb11f84"),
            "onnx/embed_tokens_q4.onnx": (857, "8773dcf4858f855bfce13def4356ca17e0bc516a477154496b6ffb6dd6d084dc"),
            "onnx/embed_tokens_q4.onnx_data": (162897920, "9210b1d26eb14136d3522584d17da14cc5f4b82b6f5607e6d80d59424a8cbe99"),
        },
    },
    "qwen3.5-2b": {
        "repo": "onnx-community/Qwen3.5-2B-ONNX",
        "revision": "e988a4ce430d7dc6fd77b4b93341b0aeae289748",
        "files": {
            "chat_template.jinja": (7755, "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80"),
            "onnx/decoder_model_merged_q4f16.onnx": (1046438, "aff64289c18b527ee16963444be405d08e7211c4fdca6f3fe53d9d4586703bf7"),
            "onnx/decoder_model_merged_q4f16.onnx_data": (1089777664, "e0d1375a8d30aa7db927ce05c60c4134345dca61150acdf05b9822bc06c7e54b"),
            "onnx/embed_tokens_q4f16.onnx": (1064, "802a072ff21f540eda7f343aa71dbb0354c8859caaf34f09b3bf8117725d7de8"),
            "onnx/embed_tokens_q4f16.onnx_data": (294010880, "650aa8eb39b7404ca2c908d78243c82b6fd88321feeb8fca175745806c6b3a81"),
            "tokenizer.json": (19226111, "89da80cc6689bef4d90cc1028249436975ffb0814618f1d93c65310e05801a9b"),
            "tokenizer_config.json": (10060, "c4cd279f8fed1ddea0d23ab13e7ec255b15fcad0200ec0487fb853d880b0b40b"),
        },
        "optional": {
            "onnx/decoder_model_merged_q4.onnx": (885982, "33f9c1311878df2140e7286f76d9e3acb29d77d91f724210674b0b63d8330df5"),
            "onnx/decoder_model_merged_q4.onnx_data": (1209126912, "c6f4807e1287961a354b0bc6f8d4c68c658f707a9541e771a297dfefaf4a1e14"),
            "onnx/embed_tokens_q4.onnx": (857, "0255dd844858758f452d9678f1e2c91db178f21b7da254d29b12e7fe3d23e305"),
            "onnx/embed_tokens_q4.onnx_data": (325795840, "9a6404d9b1c79ffc038d5c28deec04d419efb4594125453c991be38b4522ecb6"),
        },
    },
    "qwen3.5-4b": {
        "repo": "onnx-community/Qwen3.5-4B-ONNX",
        "revision": "3551704c0c04ee52b5f1bfc27d1ee8d2a94973fc",
        "files": {
            "chat_template.jinja": (7756, "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715"),
            "onnx/decoder_model_merged_q4f16.onnx": (1421099, "61b653b72f301d1ff503910d23f67ee062e41195257c874e2ef5b73e36cc00b5"),
            "onnx/decoder_model_merged_q4f16.onnx_data": (2067208192, "846fad79745d84482f8baba2d5db2d907ce42704c1b0323a3467a52288c4165b"),
            "onnx/decoder_model_merged_q4f16.onnx_data_1": (367513600, "fc1bb145d8839272a87c71e0cb4d34832a0d7bb4de06ab4fb74fea1aa6ddf7e5"),
            "onnx/embed_tokens_q4f16.onnx": (1064, "0e5fe965e5575b6428b7dea82661ed09bf7abadf29450c279e46e8113745110e"),
            "onnx/embed_tokens_q4f16.onnx_data": (367513600, "fc1bb145d8839272a87c71e0cb4d34832a0d7bb4de06ab4fb74fea1aa6ddf7e5"),
            "tokenizer.json": (19226111, "89da80cc6689bef4d90cc1028249436975ffb0814618f1d93c65310e05801a9b"),
            "tokenizer_config.json": (10061, "5d35dd3e7792dcbc1082a2f4eeb10731d78b5c7824c004c230af860b1d44e214"),
        },
        "optional": {
            "onnx/decoder_model_merged_q4.onnx": (1206737, "731f0569c84ea71a756abfe090f3b472050ecda7ad35a78fb553dc530a60315b"),
            "onnx/decoder_model_merged_q4.onnx_data": (2093368320, "c884a7c467f508568008bbaed3a90a77103f2dbc0643bd2625f8c483864c9726"),
            "onnx/decoder_model_merged_q4.onnx_data_1": (607298560, "8239304d972b0ee1661b8be4b90ccf5937073a281f62fc1917d793b1fb873f3c"),
            "onnx/embed_tokens_q4.onnx": (857, "b72feb6947db92d643f70abc6a4d3e5398a9535470a97685e3b1ab3b05047146"),
            "onnx/embed_tokens_q4.onnx_data": (407244800, "aeb939a0c9a5d54c9f3ce77db340132780692c02e602b80a0cefb6ec8a9728be"),
        },
    },
}

CACHE_ROOT = Path(__file__).resolve().parent / ".cache"


def cache_dir(model_id: str) -> Path:
    m = MODELS[model_id]
    return CACHE_ROOT / m["repo"].replace("/", "__") / m["revision"]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(model_id: str, rel: str, size: int, digest: str) -> Path:
    m = MODELS[model_id]
    dest = cache_dir(model_id) / rel
    if dest.exists() and dest.stat().st_size == size and sha256_of(dest) == digest:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/{m['repo']}/resolve/{m['revision']}/{rel}"
    print(f"downloading {model_id}/{rel} ({size / 1e6:.1f} MB)", flush=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as r, tmp.open("wb") as out:
        while chunk := r.read(1 << 20):
            out.write(chunk)
    actual_size, actual = tmp.stat().st_size, sha256_of(tmp)
    if actual_size != size or actual != digest:
        raise SystemExit(f"{rel}: expected {size} bytes sha256 {digest}, got {actual_size} bytes sha256 {actual}")
    tmp.rename(dest)
    return dest


def main() -> None:
    names = [a for a in sys.argv[1:] if not a.startswith("--")] or list(MODELS)
    for model_id in names:
        files = dict(MODELS[model_id]["files"])
        if "--with-q4" in sys.argv:
            files.update(MODELS[model_id]["optional"])
        for rel, (size, digest) in files.items():
            print(f"ok {fetch(model_id, rel, size, digest)}", flush=True)


if __name__ == "__main__":
    main()
