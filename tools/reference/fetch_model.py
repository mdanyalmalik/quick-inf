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
    # Added 2026-10-11 (WebLLM removed): more web-ready ONNX exports.
    "deepseek-r1-distill-qwen-1.5b": {
        "repo": "onnx-community/DeepSeek-R1-Distill-Qwen-1.5B-ONNX",
        "revision": "d63c4c38c538da4272c6f9ef3274196a0853b2d0",
        "files": {
            "tokenizer.json": (9117072, "0d7259f8a225ceaa421baf5ae0687e0e4a4239b86a6d1838aba50e2470d99341"),
            "tokenizer_config.json": (7053, "3ae9c54801135c7a12899efb9f684fb88a15786bd69e3ba6f714036faa3cd3a5"),
            "onnx/model_q4.onnx": (1966462264, "3c64abf7c85102af29c0c0231caf754e981faebfa83ef9530fb457d2c5140b04"),
        },
        "optional": {
        },
    },
    "qwen3-4b-instruct-2507": {
        "repo": "onnx-community/Qwen3-4B-Instruct-2507-ONNX",
        "revision": "2bdfea4417ee390519508a6bc7dfbc36f3572981",
        "files": {
            "tokenizer.json": (9117040, "e7a95fce95bf5b0946d0ddb3f9d7caa030b7e850bbe92b0edb26bcf563e9f3d5"),
            "tokenizer_config.json": (4047, "15cd319f80abfb15ebac3ee9a8c403f676b3d9a8bb44104556a207d6c67fd3dd"),
            "chat_template.jinja": (2630, "64f85b198065d0fba2a81f37e10ed68161ce2c19a754c7100e67e0ca2ee9c326"),
            "onnx/model_q4f16.onnx": (437655, "6e603b4d2324fa94a1c1fc927d1f985da78413afcffa27b02b008d909eccc4a8"),
            "onnx/model_q4f16.onnx_data": (2094347264, "d2b17463255a120a6d4e88c60e24bb6be6607ab6e34dd89b62234879bfd6ce0d"),
            "onnx/model_q4f16.onnx_data_1": (794787840, "6c0d3b7ee94fa08c2dc766dfa4c5b790e492077701847b40fa001f648f3ebded"),
        },
        "optional": {
        },
    },
    "gemma-3-4b-it": {
        "repo": "onnx-community/gemma-3-4b-it-ONNX",
        "revision": "944e625cdc227b1f1c41db1ae6bba655cda90104",
        "files": {
            "tokenizer.json": (20323105, "b472c7c2d1a2ac607daa53c794575e59504c06b39579a0f2399bb95cea6a4949"),
            "tokenizer_config.json": (2327, "362055c2b4c69af56db94be4f1297bc6d2936a8f4ab9c0173b5bedd502280507"),
            "chat_template.jinja": (1532, "7de1c58e208eda46e9c7f86397df37ec49883aeece39fb961e0a6b24088dd3c4"),
            "onnx/decoder_model_merged_q4.onnx": (467689, "70269ef03c0cdb9c1f247c4ac76e75036ee7df09757b9ac165308fd80738df71"),
            "onnx/decoder_model_merged_q4.onnx_data": (2093692928, "4d339ace22e320b447a52e7861367328efd4a4b742920ed0bab0c0631e99b46c"),
            "onnx/decoder_model_merged_q4.onnx_data_1": (661772800, "c854198bff7eeac9948f675148f3b4f0582e433f9bc1326a3d3af55cf82cbb02"),
            "onnx/embed_tokens_q4.onnx": (858, "152862f7d988a855514daa3e88643cd15288f88a395930e69bf765a5679e8ccf"),
            "onnx/embed_tokens_q4.onnx_data": (430021120, "6b137a3b8184478ffd73c2e9e9af38874f9dfd8a16401cc095093dc10fef5896"),
        },
        "optional": {
        },
    },
    "smollm2-135m-instruct": {
        "repo": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "revision": "12fd25f77366fa6b3b4b768ec3050bf629380bac",
        "files": {
            "tokenizer.json": (2104556, "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c"),
            "tokenizer_config.json": (3764, "4ec77d44f62efeb38d7e044a1db318f6a939438425312dfa333b8382dbad98df"),
            "onnx/model_q4f16.onnx": (117691126, "9358cd4ce037c304621f8c194a525607ae7c5ea73239fcae4c21bd02f2e34ff7"),
        },
        "optional": {
            "onnx/model_q4.onnx": (182068553, "933577110303a2964096d19b6f15d3b4639bef7f99481ac0b61d9f3ad72f392a"),
        },
    },
    "gemma-3-270m-it": {
        "repo": "onnx-community/gemma-3-270m-it-ONNX",
        "revision": "2dbbfdb1b59bd034eb959428c6a7da9dd7ea27f0",
        "files": {
            "tokenizer.json": (20323013, "55da1312bdf1d7d8fe8d9d1b3eed04086261149e6034e0ac3f8c633b67f5aac8"),
            "tokenizer_config.json": (2313, "8ae5c9297e0ca85077c0f4952f85251b4773af7abc4673b62b54bc0b609a9667"),
            "chat_template.jinja": (1532, "7de1c58e208eda46e9c7f86397df37ec49883aeece39fb961e0a6b24088dd3c4"),
            "onnx/model_q4f16.onnx": (330717, "01d620bf3c2c1d5b843a5c92a4456ce1f4c63f6ceb814da26a3420e8aad2b4a4"),
            "onnx/model_q4f16.onnx_data": (272626176, "23088037c5972441c6a6033f5fdcca41c96bfea9b0bf7e1abd6038ba42e45287"),
        },
        "optional": {
            "onnx/model_q4.onnx": (242522, "cb3615746a988839308507933d4e0501ff695e747a0a0ce15c0ef53083f8ec54"),
            "onnx/model_q4.onnx_data": (322933248, "25c4a14af32fc1011a90504076024b0e1f65f915e9690b44a5484200d2e20851"),
        },
    },
    "lfm2.5-350m": {
        "repo": "LiquidAI/LFM2.5-350M-ONNX",
        "revision": "d11593fd9eb408e322667926656598896c2d5ff9",
        "files": {
            "tokenizer.json": (3297793, "29d43b4be8e8a896fefd7cd836ca6d6b4eedd249f823866ce0453b368e646f49"),
            "tokenizer_config.json": (3269, "95c85d0860d06c9529345f386004e8e67743375b15c5d39e9f46427d8977577b"),
            "chat_template.jinja": (2553, "013eed60546434b6967e3483153d8c5c37abcb1d667f8b1f914683f2a9411531"),
            "onnx/model_q4f16.onnx": (182827, "3012c28a119828561c90196331435d91c43c1a6ab4898ac79715585fff1dff85"),
            "onnx/model_q4f16.onnx_data": (254965760, "9256ecd417b801b441d926ebe5ead4ebf50ac1a5a0f8f7914cf60d9d6452ef69"),
        },
        "optional": {
            "onnx/model_q4.onnx": (183442, "d1a705712e93aafaba1346b32245fa59e7857a46e2272003c0e8c524977e0de8"),
            "onnx/model_q4.onnx_data": (293629952, "71ec6ad38a4c463dcb3dba671d06a1d9861be3a23e51290d818b95c0b7d2a5db"),
        },
    },
    "granite-4.0-350m": {
        "repo": "onnx-community/granite-4.0-350m-ONNX-web",
        "revision": "62fc8f345c8aed0e1a28f856dc7e79b4cdb55cae",
        "files": {
            "tokenizer.json": (4131199, "ea076a8bccab905484daa97acc7021a915fe30c1ad48fa85bd4b82a5d1ceb797"),
            "tokenizer_config.json": (24747, "36d15e9ab9ed50ac434fcfd2620b829615c80d745f36a98f2454620a3fc395c5"),
            "chat_template.jinja": (6418, "9524df67b77a7b25a2dfee898f75b316a157eb9d855b51e32aeac79d7c8a83ce"),
            "onnx/model_q4.onnx": (272846, "17e171e1d30f1c644eda76d673e662996ff6b89cdb22a11e2ecadde9c158535f"),
            "onnx/model_q4.onnx_data": (575639552, "c322cb280cd7168e37d5d187656973ac7ccecc1465c29f52c5547b2f6e557799"),
        },
        "optional": {
        },
    },
    "qwen2.5-coder-0.5b-instruct": {
        "repo": "onnx-community/Qwen2.5-Coder-0.5B-Instruct",
        "revision": "604bb1988ff789d5024d589c63cf9ae81c0c8ab5",
        "files": {
            "tokenizer.json": (11421896, "9c5ae00e602b8860cbd784ba82a8aa14e8feecec692e7076590d014d7b7fdafa"),
            "tokenizer_config.json": (7849, "6a14243f0ae04884e597b3f032c05e15de9f417318fdcac24ab4e2680e691430"),
            "onnx/model_q4f16.onnx": (554935833, "60c076ac0d3910881fe0cad75997e803459a5f71bb74bcbcd82407ef83db7ba0"),
        },
        "optional": {
            "onnx/model_q4.onnx": (862087110, "2df7ea9739c3c26bd9d2cad0815768a5a0a985d4b5548b7de80c5e85e9a97b47"),
        },
    },
    "lfm2-700m": {
        "repo": "onnx-community/LFM2-700M-ONNX",
        "revision": "bdfeecd9300145cc1696702b74234d73f683622f",
        "files": {
            "tokenizer.json": (3296940, "bec04fe03d27dbcb6276b46e7d2325433323a8132a2f0404ea6bbf98250a9211"),
            "tokenizer_config.json": (93557, "1635881015705807ba0fb6510d77a42e53fc041f4de12bb801f1685a62b7220d"),
            "chat_template.jinja": (1349, "2294f49b51a0b9a1b0194add3f124237b5bd986599f43180eec49bba4de01365"),
            "onnx/model_q4f16.onnx": (183018, "95ce2f28ec97b145913e7b474209c36bf5a174916fe6a13fd205531db7e24bcf"),
            "onnx/model_q4f16.onnx_data": (496104448, "b0cd094b1731ce79e5bcc140f874147b1dc4554248ebcf6ed27c0126ad99776a"),
        },
        "optional": {
            "onnx/model_q4.onnx": (183625, "496549029747546240c69a5d11221ef3edce8c6d6f41051c42244ba47aedcefb"),
            "onnx/model_q4.onnx_data": (559081472, "c063f8b36e0aa87cf659cd8c15ac23178fc90a2390f708d556d1d12d59cd2715"),
        },
    },
    "gemma-3-1b-it": {
        "repo": "onnx-community/gemma-3-1b-it-ONNX",
        "revision": "a58439f40017d3b99c7d378ff525e54e0ba08ebf",
        "files": {
            "tokenizer.json": (20323013, "55da1312bdf1d7d8fe8d9d1b3eed04086261149e6034e0ac3f8c633b67f5aac8"),
            "tokenizer_config.json": (2327, "362055c2b4c69af56db94be4f1297bc6d2936a8f4ab9c0173b5bedd502280507"),
            "chat_template.jinja": (1532, "7de1c58e208eda46e9c7f86397df37ec49883aeece39fb961e0a6b24088dd3c4"),
            "onnx/model_q4f16.onnx": (461341, "522d1f0e1d1fd0065ca02d61a0bf17a43701c1835b4ebf0c4628b73f1a6fb796"),
            "onnx/model_q4f16.onnx_data": (763067904, "6fb9bc8dd4b93a040bcea60c953c2e24f60f9d52e708befec077f6ab60baa007"),
        },
        "optional": {
            "onnx/model_q4.onnx": (347363, "69686023e5892376e38fcbcdd0c77af432c55b3bcd03aee6d561bd1f04507da0"),
            "onnx/model_q4.onnx_data": (859106816, "c2370070be257a98d50e17d81be13e18304c39e7e6d9d1416f8f883681d2a17b"),
        },
    },
    "granite-4.0-1b": {
        "repo": "onnx-community/granite-4.0-1b-ONNX-web",
        "revision": "d65b1a480bfab70bc828208d7903cf9de4bdfaa1",
        "files": {
            "tokenizer.json": (4131199, "ea076a8bccab905484daa97acc7021a915fe30c1ad48fa85bd4b82a5d1ceb797"),
            "tokenizer_config.json": (24747, "36d15e9ab9ed50ac434fcfd2620b829615c80d745f36a98f2454620a3fc395c5"),
            "chat_template.jinja": (6418, "9524df67b77a7b25a2dfee898f75b316a157eb9d855b51e32aeac79d7c8a83ce"),
            "onnx/model_q4.onnx": (389658, "4e125c4855f430c70581f457d6d5d21803dca366b628cf8931b9708c480e0523"),
            "onnx/model_q4.onnx_data": (1781145600, "3d32a0c75d82304bacf353b2c531e2dbb958e51286cfb426123e6d4918a9a92c"),
        },
        "optional": {
        },
    },
    "lfm2.5-1.2b-instruct": {
        "repo": "LiquidAI/LFM2.5-1.2B-Instruct-ONNX",
        "revision": "10f72e70abf67ac0fd7ebf15bc5854726891d864",
        "files": {
            "tokenizer.json": (3297799, "0a1f2cb9fc2769030ca1f4207e81c1af493319373d41b8d3cf5be53860c13760"),
            "tokenizer_config.json": (2391, "0f11f4ffa5750369414bb24e41f26e2bbf23f1d01def533cad5f584e59ae3bef"),
            "chat_template.jinja": (1783, "f05bf4b967dc993bdc7a2fe6e43759ee218eb0eb340d68b063e1c4f8ad148176"),
            "onnx/model_q4f16.onnx": (182795, "a9986ad188200507342ac32f727aa4691edb5428b0aa8c4a9fbdf0c85a6fe667"),
            "onnx/model_q4f16.onnx_data": (760279040, "46cfacc12941150620a3f644a5269e9baebd75d681cfa09cadeef71b8ed64ac2"),
        },
        "optional": {
            "onnx/model_q4.onnx": (183173, "c760a2d8be1d19c71c470ef294c1220ef85b5b3a0e31f12a363655ea7fab1d58"),
            "onnx/model_q4.onnx_data": (850059264, "d9666c44e2acc32f06c9351f9e7c4fd66bc060d88ca8e7f954e836c6845f7488"),
        },
    },
    "qwen2.5-math-1.5b-instruct": {
        "repo": "onnx-community/Qwen2.5-Math-1.5B-Instruct",
        "revision": "29b745e87b28faa666ea598ee43b4db439144be6",
        "files": {
            "tokenizer.json": (7031673, "a8506e7111b80c6d8635951a02eab0f4e1a8e4e5772da83846579e97b16f61bf"),
            "tokenizer_config.json": (7316, "1a4b7678803ae074740c378b8de4d7ef22e3f6b436ccdd8a3a9ebea5d34c9dcf"),
            "onnx/model_q4.onnx": (1758206404, "c00fb4d8c11c5888845408916d126d2c636c2c4f0f4b96ed4bb59531a79c5594"),
        },
        "optional": {
        },
    },
    "lfm2-2.6b": {
        "repo": "onnx-community/LFM2-2.6B-ONNX",
        "revision": "cd7301a675f9b6eab8c789488d9b22333b2fd023",
        "files": {
            "tokenizer.json": (3296920, "0d97ddf099eae6c0b4392bdffa1256762eb8621de54e04429e0719965036c3d7"),
            "tokenizer_config.json": (93603, "36fc046d9b6b17c3065c479110871afb7132ecb814a008fc2382a1533d2a0f1d"),
            "chat_template.jinja": (1296, "89e790f027916b5a2bca145a6a8454e06ffc7a5043bf3b6d97829aff86bb543f"),
            "onnx/model_q4f16.onnx": (322750, "d98c4f6cf16c142f928127c01be89b0199d7a0c8223c60a756bfde21afe33878"),
            "onnx/model_q4f16.onnx_data": (1654910976, "3dd89d13c6c716bba3fe196f9b6cd22abc51a1a476efb70a8c0151b14a5f8cfe"),
        },
        "optional": {
            "onnx/model_q4.onnx": (323506, "26cecca7cc85cad6d58766a1081d899e7cd1388af7c59afdd5c0d74da3d20429"),
            "onnx/model_q4.onnx_data": (2092425216, "5f80e6c5eb5787c0c5825b8772f614113b09e6ac23be4dba58c8447c963bf157"),
        },
    },
    "qwen2.5-coder-3b-instruct": {
        "repo": "onnx-community/Qwen2.5-Coder-3B-Instruct",
        "revision": "0e076a9c5291f8c74b3ce1558f21336b18f94739",
        "files": {
            "tokenizer.json": (11421896, "9c5ae00e602b8860cbd784ba82a8aa14e8feecec692e7076590d014d7b7fdafa"),
            "tokenizer_config.json": (7849, "6a14243f0ae04884e597b3f032c05e15de9f417318fdcac24ab4e2680e691430"),
            "onnx/model_q4f16.onnx": (271268174, "003639a5dd89da951693ac9ec29c16ff1c557d6669bf830cca9eb4546cce100f"),
            "onnx/model_q4f16.onnx_data": (2096365568, "2216b89a17ab28efcf5700446a870464ee418fec2f5388198bb7a107c2cf0288"),
        },
        "optional": {
            "onnx/model_q4.onnx": (1098040858, "cf4854a472f6a7b800a1a9967d8a6faa5dd451fb4b012ac8c5712daf2959610e"),
            "onnx/model_q4.onnx_data": (2093154304, "cd6ce254e9d59e14e015fbb9725d018c8d8bda3588abe72c31f4bb5e2aefa68b"),
        },
    },
    "granite-4.0-micro": {
        "repo": "onnx-community/granite-4.0-micro-ONNX-web",
        "revision": "055c9bd9db8f5e47bebd17246b0c614ffe5e8a13",
        "files": {
            "tokenizer.json": (4131199, "ea076a8bccab905484daa97acc7021a915fe30c1ad48fa85bd4b82a5d1ceb797"),
            "tokenizer_config.json": (24423, "c7ad428bdfe33e40bd5d56fc5032f8916f10a38ce0d80b4919cd3e32f44fc808"),
            "chat_template.jinja": (6099, "fed2756d2d24e127b951dcf139d0b03ab7db8ef23a456128ebc9c2db4901d476"),
            "onnx/model_q4f16.onnx": (391777, "c261780a57c22ff7b9107facea0c035a3be0501248e81da6da96a35a13ad62c0"),
            "onnx/model_q4f16.onnx_data": (2088129536, "847d3496fd4d6a2cf70c5dd5920fa50b47754f0c963c8cec92563508801d1eb7"),
            "onnx/model_q4f16.onnx_data_1": (212336640, "4863091011d2cceb4325f5d0e7eac90c28b602566951cce8742ce67ee8442d98"),
        },
        "optional": {
        },
    },
    "phi-3-mini-4k-instruct": {
        "repo": "microsoft/Phi-3-mini-4k-instruct-onnx-web",
        "revision": "80a2792f5bf861528ce9b449b3230f1bd3fdc759",
        "files": {
            "tokenizer.json": (1937870, "8e9e94300b0c19471520eca5e6cacc4a522c7da823edac3b2cdd1152b4fb3ae2"),
            "tokenizer_config.json": (3299, "5e1bcaa42cf573a3ae063f8bb2d54bd5aaf8150155c4670b29aa2caa012ef3b1"),
            "onnx/model_q4f16.onnx": (838080425, "e726f1c300be9714987ebf7cb047afcb002f6ce9a28af1771d05d5d1d4eeaf48"),
            "onnx/model_q4f16.onnx_data": (1454211072, "62eb6a6870961b0039732d574779e44a7de42e26c04ca1433759728d644cc00f"),
        },
        "optional": {
            "onnx/model_q4.onnx": (1059602332, "16b8e5d28a757c37bbfa7d9420fd094c0c20e3615ca3c203b5b9501015045c8f"),
            "onnx/model_q4.onnx_data": (1663524864, "41d30b87f06b52e6b24c4e2e65a6a14e5c9fb5bc6f495fac17b19c6bc7875ff5"),
        },
    },
    "olmo-3-7b-instruct": {
        "repo": "onnx-community/Olmo-3-7B-Instruct-ONNX",
        "revision": "64cf5a8d2dac83c3b889d44c422de2ae3e2b2e7d",
        "files": {
            "tokenizer.json": (4119438, "e6297cf3c369c31fc24313131019d0cf6d7ddd137ea1e75d7a65eca7edfcd522"),
            "tokenizer_config.json": (3085, "8c5ed5d8353c70677464e7e09da5771f9776d6f3b43358ba4756ba587ab449cb"),
            "chat_template.jinja": (2613, "f5186d42d99c8a0445d37fd8a6c7ccf07fe3e24a29ce622d8bd245da9507b12b"),
            "onnx/model_q4f16.onnx": (414480, "26f2b043f4567287ac00c7669e38b1356a5feb7ae5b2ab37a5e91ed6dc6fc65f"),
            "onnx/model_q4f16.onnx_data": (2077432032, "32aeeba1ada2d4777d5b5e0b3b37bb334644e53ab021f846a15ea6f5f7aaccc1"),
            "onnx/model_q4f16.onnx_data_1": (1732204768, "39c21ba9fd832cc978da7670a01a865f83998fced33c0b4e41b7700e7348bbb3"),
        },
        "optional": {
            "onnx/model_q4.onnx": (363850, "b84a33e314a2a0ac7e9a0345047beb1a84f131978dcc58dc8593c5998980daee"),
            "onnx/model_q4.onnx_data": (2096617056, "881451b57726b3eda20f9a2f03dc2edee17287224b7311f11155e09941925971"),
            "onnx/model_q4.onnx_data_1": (1844344416, "e78bc82239d56f6e91dc959539bba406e79c9cef05eccf32767b184308f2e394"),
        },
    },
    "lfm2-8b-a1b": {
        "repo": "onnx-community/LFM2-8B-A1B-ONNX",
        "revision": "c9bf5d746af56d087076debc9d0b59f73434d8c8",
        "files": {
            "tokenizer.json": (3297530, "40b989ce8875ab2d4be98959e764b4dc761b8d3bfd1dcfe48f8d69f559eac3f4"),
            "tokenizer_config.json": (2489, "3c07171345a9d530c1cd36526111e343f6545eaccd95686de88a1a88afba04fa"),
            "chat_template.jinja": (1296, "89e790f027916b5a2bca145a6a8454e06ffc7a5043bf3b6d97829aff86bb543f"),
            "onnx/model_q4f16.onnx": (298018, "fe3ac00b2877bfa4a18aba49971d90c41a53af250ce258636be6f164ecd9ee2d"),
            "onnx/model_q4f16.onnx_data": (2076209152, "d2aa8b7ae553d6acc3af770b4f7624dccf6cce3fb0a0c8e417afe162ccd1ad05"),
            "onnx/model_q4f16.onnx_data_1": (1981808640, "77de2b5d304514a38f8ffd9a08cfd1fe1fc810202e83bc55df76e9a8551ce4a1"),
            "onnx/model_q4f16.onnx_data_2": (726185984, "462d873930e717de9d2f106e60ca2ddb62458360139afdc528c873f2b0f3e33e"),
        },
        "optional": {
            "onnx/model_q4.onnx": (300329, "99b1c373c7d1c5cbe71645992f620540ffa24542bdea2b80dd6c6843454e3ab3"),
            "onnx/model_q4.onnx_data": (2086723584, "1e5ee60e6cbaf6d70ffd000fff2667a0b2e4ffd454e0f90ae05edcc617745caf"),
            "onnx/model_q4.onnx_data_1": (1996488704, "5d3846396d300289e7813da521ed9fc2df9fe4012a773d678daf447705f12d10"),
            "onnx/model_q4.onnx_data_2": (1239029760, "d0311083ed54febf6439bd72ee115961c4235f3e1c6b3d9ded34c08b640f6ced"),
        },
    },
    "lfm2-24b-a2b": {
        "repo": "LiquidAI/LFM2-24B-A2B-ONNX",
        "revision": "e6fb57d2e0370252b077858efcdd906d1c4b08df",
        "files": {
            "tokenizer.json": (4733389, "df1d8d5ec5d091b460562ffd545e4a5e91d17d4a0db7ebe733be34ed374377bd"),
            "tokenizer_config.json": (3238, "905f6a8cf2a628581b7a6393e72759491e0d5d48640bbbf1a4ad5e092645f0e1"),
            "chat_template.jinja": (2553, "013eed60546434b6967e3483153d8c5c37abcb1d667f8b1f914683f2a9411531"),
            "onnx/model_q4f16.onnx": (495227, "18725e7bab28d23cf67a01718909c1d99e79701d967a3990f20241316c5e60fb"),
            "onnx/model_q4f16.onnx_data": (2019864576, "12f9ad0a641420cf6afb9c4ddae3c0efff3ad53e25a95859bd615416a74a34e6"),
            "onnx/model_q4f16.onnx_data_1": (2038431744, "ae5a54829eadaa5802d946fcda7a7961884079c3441b8c2f5ddce4367349064c"),
            "onnx/model_q4f16.onnx_data_2": (2038431744, "868144536237c0556720922cf223c6f91a610be33e7b97b900000422781a43ae"),
            "onnx/model_q4f16.onnx_data_3": (2038431744, "849719320cc78d6c6bf23c75eade5cc16079a88d86563656e1fa79d6941614b7"),
            "onnx/model_q4f16.onnx_data_4": (2038431744, "aa7b9076059dbd97321ccfb74712e55264d8e4aad251392106c678b5553773d0"),
            "onnx/model_q4f16.onnx_data_5": (2038431744, "b82d51164169742513933204011beb2bdbf7d7ad7e5da2a1d1984aae2d0ae4d0"),
            "onnx/model_q4f16.onnx_data_6": (1297969152, "b301737d179d7142ac10bfa9113496f51d8ca9ce1effd6d5de2834c542ca3894"),
        },
        "optional": {
        },
    },
    "lfm2.5-24b-a2b": {
        "repo": "onnx-community/LFM2.5-24B-A2B-ONNX",
        "revision": "2357244129d55b9338cdd996a15b65b50ec995ad",
        "files": {
            "tokenizer.json": (17905598, "695be7802a0e4b8a81048f0ff5ebb7fc811a0ba5a6be63dbb24deb5a81096f41"),
            "tokenizer_config.json": (6101, "6cbf3cffb84336a098a3e0d31eeb4f2b1df8478bbd4760638be1cded75b27eb4"),
            "chat_template.jinja": (5443, "ea663864491de7ade391839479860ca95541f892f72665c73251fbd4643b1bef"),
            "onnx/model_q4f16.onnx": (419720, "aede747e6ac6bd35cc7c98620df59f3da2cd8f5e454ad95b4eef568e8a14fba2"),
            "onnx/model_q4f16.onnx_data": (2094215168, "fe0327b1f2aaaf93e8c0b89b911feca41c4eb283b3f89d4c6cba387d9e3656f9"),
            "onnx/model_q4f16.onnx_data_1": (2038431744, "4fe12c18d6451511d8df38510e4ca9a8b69220bc3cfcc602f00d8069f33c07d6"),
            "onnx/model_q4f16.onnx_data_2": (2038431744, "7072ff94612ac029f4a8dee1f888773dc3c4f981b134d146eaea4f5337f5b3b3"),
            "onnx/model_q4f16.onnx_data_3": (2038431744, "796c857aeea52c80459d9f885619cdb1131352bf023cf470c76b797d539a90c7"),
            "onnx/model_q4f16.onnx_data_4": (2038431744, "859cb707ab9bf6292d0d5608ceeaa822ed1a7e155fd45c91bc525f650dd98428"),
            "onnx/model_q4f16.onnx_data_5": (2038431744, "95f788177114dac016803b631fc6e375bc64c387dc9c9622c03c62940f11b95f"),
            "onnx/model_q4f16.onnx_data_6": (1228369920, "637f5e016049a22713947ed67f9e29713fa4d342f20839b6814d5e56d82aadf3"),
        },
        "optional": {
        },
    },
    "phi-4-mini-reasoning": {
        "repo": "microsoft/Phi-4-mini-reasoning-onnx",
        "revision": "8d5c831bede5ea4d224a9c82f1b30ddee9a40dd4",
        "files": {
            "gpu/gpu-int4-rtn-block-32/tokenizer.json": (15524195, "166c68da2b6ef97ff617410fc0dc55f422b33be8c2c2409c8fa13ba137d84c61"),
            "gpu/gpu-int4-rtn-block-32/tokenizer_config.json": (3216, "16d7cf2abc1139ffc61368dbbacd521d1be29baa30e461382f3bea947cba16ec"),
            "gpu/gpu-int4-rtn-block-32/model.onnx": (287586, "59ab70c351e50301c1cf4cca3f23176bba73e43f0d94cea26d471ba43b898f65"),
            "gpu/gpu-int4-rtn-block-32/model.onnx.data": (3413194752, "5fc3832908b14538d2d53597507a10160e1307c45e153fa5822d82e6248471c3"),
        },
        "optional": {},
    },
    "mistral-7b-instruct-v0.2": {
        "repo": "microsoft/mistral-7b-instruct-v0.2-ONNX",
        "revision": "6d524105ea40470ca1897c97cc467f1f96a8a077",
        "files": {
            "onnx/directml/mistralai_Mistral-7B-Instruct-v0.2/tokenizer.json": (1795303, "fc4f0bd70b3709312d9d1d9e5ba674794b6bc5abc17429897a540f93882f25fc"),
            "onnx/directml/mistralai_Mistral-7B-Instruct-v0.2/tokenizer_config.json": (1460, "0b9444de796ce8352bc4c235aa6fc6ca43c63baf98015cdba16bc8b5ad296c88"),
            "onnx/directml/mistralai_Mistral-7B-Instruct-v0.2/model.onnx": (11999645, "298f8cc3cf1b3dfeddcc3912a1900f235499fb3e4354389cbbf08fac99666e55"),
            "onnx/directml/mistralai_Mistral-7B-Instruct-v0.2/model.onnx.data": (3956748288, "ca1244eb774db3cc813d308bb56e6671387c33380fa9145a61fdf40cc62df891"),
        },
        "optional": {},
    },
    "llama-3.1-8b-instruct": {
        "repo": "onnx-community/Meta-Llama-3.1-8B-Instruct-ONNX-DirectML-GenAI-INT4",
        "revision": "aaaabf87557f7aa68b3100638fe69a9c9cc62290",
        "files": {
            "tokenizer.json": (17209920, "6b9e4e7fb171f92fd137b777cc2714bf87d11576700a1dcd7a399e7bbe39537b"),
            "tokenizer_config.json": (55381, "f0f2d5fa9caf736f4184e3fe3316378290696f9ebd9f11ac431d48fbda92a11c"),
            "model.onnx": (284037, "a17089426f1d609ca4c53b50358b3f87eb37626e342500b7a5d746c97e8255a7"),
            "model.onnx.data": (5306130432, "bf16e53214a520b52d4e5d43b64986fccfd6cc6b90cd98447e0b81980dfccc03"),
        },
        "optional": {},
    },
    "deepseek-r1-distill-llama-8b": {
        "repo": "onnx-community/DeepSeek-R1-Distill-Llama-8B-ONNX-DirectML-GenAI-INT4",
        "revision": "eb08877000b8ba75a7097c56acdf58cc49ab077d",
        "files": {
            "tokenizer.json": (17209530, "d91915040cfac999d8c55f4b5bc6e67367c065e3a7a4e4b9438ce1f256addd86"),
            "tokenizer_config.json": (52919, "a8d8c110163d3ef9d1084f9ce2bab453fd3000d8663b0efc0c19aced26cbe286"),
            "model.onnx": (284037, "a17089426f1d609ca4c53b50358b3f87eb37626e342500b7a5d746c97e8255a7"),
            "model.onnx.data": (5306130432, "2e8fcf924a0ae90140c0099f6c55d5aa8a8259810185afc2e2c33b0a527c488f"),
        },
        "optional": {},
    },
    "gemma-2-9b-it": {
        "repo": "onnx-community/gemma-2-9b-it-ONNX-DirectML-GenAI-INT4",
        "revision": "f0105d176397fc093f101fb73b937d2e90571d9d",
        "files": {
            "tokenizer.json": (34362873, "5f7eee611703c5ce5d1eee32d9cdcfe465647b8aff0c1dfb3bed7ad7dbb05060"),
            "tokenizer_config.json": (47026, "80838cecd2695584b0f73177cffb7f802573cb6bf009f2e57651d12deb528b54"),
            "model.onnx": (411237, "7ef9b72f5b8d2333a60e42a229b02879062171776ba4e135b37ce6dd394f9114"),
            "model.onnx.data": (7038532608, "8fde2299ee288fc1a49ed9a45e28aea844dd590d2ae2d5f8580d0bd0b2c93813"),
        },
        "optional": {},
    },
    "phi-3-medium-4k-instruct": {
        "repo": "microsoft/Phi-3-medium-4k-instruct-onnx-directml",
        "revision": "6e74e8bd07ed934dc7521e1e9b4c1a7121284193",
        "files": {
            "directml-int4-awq-block-128/tokenizer.json": (1844436, "d0f067e1e15cd0a36ebef3668024882cb67a80b86fb4b7b4b128481f0d474db7"),
            "directml-int4-awq-block-128/tokenizer_config.json": (3155, "7a176e2c0b8aa486b71a86d94bfd2a298ead18728becf25e5b69f436f2b98bd5"),
            "directml-int4-awq-block-128/model.onnx": (3750885, "ec36b45ae6c94ff49935535b6c78e591e15d6fb5bf9d3e1e9e1f03da03a32285"),
            "directml-int4-awq-block-128/model.onnx.data": (7496439040, "d5bdb5e59d5bde62f6ac2bf0c8969d61d80ae6bffc216b171c4105b6271c382d"),
        },
        "optional": {},
    },
    "phi-4": {
        "repo": "microsoft/phi-4-onnx",
        "revision": "c4e245b57793e8481e0fa9252b3a45dee44a921b",
        "files": {
            "gpu/gpu-int4-rtn-block-32/tokenizer.json": (7153083, "c612e57b2763204798271c0cdc14a373a565e6a371f1bfbd44e385a8391c9884"),
            "gpu/gpu-int4-rtn-block-32/tokenizer_config.json": (17714, "0eb6d6c0b18f0730c9714dd429004c66aac14b59cd05ee88b9f825a24c12f575"),
            "gpu/gpu-int4-rtn-block-32/model.onnx": (358825, "94ce65a52a4ef7fcea28bb5e5a677bf57968116f68c50a5b6ba63e4439fe85c1"),
            "gpu/gpu-int4-rtn-block-32/model.onnx.data": (8989353984, "af8d99d133c18ac59cc84395cb8ee2a82e753c4662366a40c69ea34ff561fa19"),
        },
        "optional": {},
    },
    "phi-4-reasoning": {
        "repo": "microsoft/Phi-4-reasoning-onnx",
        "revision": "f5d8c9c1f0dad99f7d6212eee2c8ea4ebee7b935",
        "files": {
            "gpu/gpu-int4-rtn-block-32/tokenizer.json": (7153070, "b28a32dcbbb2779573aa8e345457e7b10f470ccfd262d01672c2469698cbd4f7"),
            "gpu/gpu-int4-rtn-block-32/tokenizer_config.json": (19017, "889e1696cfd3ac5f091e676b9ea78d8899a1f45d18a2f0f96bb7221d2b172a80"),
            "gpu/gpu-int4-rtn-block-32/model.onnx": (358825, "bc2038958d9a92310c4fc3727bd51b760bc6aa915bd8d9c897f61070ddfdfc93"),
            "gpu/gpu-int4-rtn-block-32/model.onnx.data": (8993548288, "9d2e92e8d7d3cbe31e8613a1f8f884eafbae27bf3bcf2bf9646408665f21fa8b"),
        },
        "optional": {},
    },
    "tinyllama-1.1b-chat": {
        "repo": "Xenova/TinyLlama-1.1B-Chat-v1.0",
        "revision": "2d31d778084d99c2f58d14adfaa7a27f100d9391",
        "files": {
            "tokenizer_config.json": (1340, "5030b57187ef621d5e9f63cdd2e3531a9004f579326c0df62bc15ca9107582c3"),
            "tokenizer.json": (1842767, "bcd04f0eadf90287bd26e1a183ac487d8a141b09b06aecb7725bbdd343640f2e"),
            "onnx/model_q4f16.onnx": (713747554, "a6281a71cb9c6a84390a9ecfc1dcd255624908421e667bfa3527dd955a7fb93d"),
        },
        "optional": {},
    },
    "qwen3-8b": {
        "repo": "onnx-community/Qwen3-8B-ONNX",
        "revision": "eab7c5b1ac4637f6745e5e1d21ffa0bec4228186",
        "files": {
            "onnxruntime/webgpu/webgpu-int4-kld-block-32/tokenizer_config.json": (663, "bfd13b57b2e0c2cb582f311c0895f077d16a868c0dc61e4dbbba544a835341d6"),
            "onnxruntime/webgpu/webgpu-int4-kld-block-32/tokenizer.json": (11422648, "979d160e081df25a1bf7f4e2e8f4c441b5dfdc9a8e84aec9f32e80445e1b59b8"),
            "onnxruntime/webgpu/webgpu-int4-kld-block-32/model.onnx": (487229, "dd11227b49ff020600ef9cb52cfd5e6f62cadb191b53dfc1975c87d9477206b9"),
            "onnxruntime/webgpu/webgpu-int4-kld-block-32/model.onnx.data": (6434848768, "9fc6adbf230c2187570ab843a9b05d480cd317a6f74422bc26af621b87080df5"),
        },
        "optional": {},
    },
    "qwen3-14b": {
        "repo": "onnx-community/Qwen3-14B-ONNX",
        "revision": "8a58d280ce227fe9bcbcff80704f957dd6cb70f0",
        "files": {
            "onnxruntime/webgpu/webgpu-int4-kld-block-32/tokenizer_config.json": (663, "bfd13b57b2e0c2cb582f311c0895f077d16a868c0dc61e4dbbba544a835341d6"),
            "onnxruntime/webgpu/webgpu-int4-kld-block-32/tokenizer.json": (11422648, "979d160e081df25a1bf7f4e2e8f4c441b5dfdc9a8e84aec9f32e80445e1b59b8"),
            "onnxruntime/webgpu/webgpu-int4-kld-block-32/model.onnx": (542103, "6f3f780f00f5e5f6202f2ad25c11c670c3c9959c276f46d4c182b2ca0ddbb99e"),
            "onnxruntime/webgpu/webgpu-int4-kld-block-32/model.onnx.data": (10654449664, "9885798b8f1c5bb1bc56d55742abfd5b7456ad602f669441e6ca58368c248bfe"),
        },
        "optional": {},
    },
    "deepseek-r1-distill-qwen-14b": {
        "repo": "lokinfey/DeepSeek-R1-Distill-Qwen-14B-ONNX-INT4-GPU",
        "revision": "a13623d729b3829d3d10a723deb2f48ac22adc48",
        "files": {
            "deepseek/tokenizer_config.json": (3071, "8ac8c85fb242563c2260baec0909debd69d718af6a0b3d90e6cab62b4d341cd5"),
            "deepseek/tokenizer.json": (7031660, "88145e3c3249adc2546ede277e9819d6e405e19072456e4b521cbc724bd60773"),
            "model.onnx": (336289, "d2cedf30bcbd6e583edbfb3e4a2041f24d8366a5bc205e1656bed488990f1691"),
            "model.onnx.data": (9462097920, "17a1fb577311ee775d3badc28197bbfb8e3e1b76962ccf3b39eb6f1f0e3202b1"),
        },
        "optional": {},
    },
    "ministral-3-3b": {
        "repo": "mistralai/Ministral-3-3B-Instruct-2512-ONNX",
        "revision": "e1128d389145f96bb1c3c2625cb8ca255ffa7034",
        "files": {
            "tokenizer_config.json": (206014, "8e0096c9c11500fd6230c38408ad53fd926ab2408057bf3421839cfd6082bc21"),
            "chat_template.jinja": (7753, "d28d7df94f0fd7e8d0075a22c473333d6e7dd2bc4c36c83e8b975300a0fb94bc"),
            "tokenizer.json": (12603042, "fa0cdf170403fdfe173ebb7356b0443d3f3908aeed183d24712a0410631540f2"),
            "onnx/decoder_model_merged_q4f16.onnx": (281991, "9e591109067dcc9efc1b14daaa6e0e78ed917f03ff242a3a1bf4e3bc13478a56"),
            "onnx/decoder_model_merged_q4f16.onnx_data": (1996158976, "d5985999bc1755741860a1cf4838549f381bc5d32eb09379add5ee60f0282dd2"),
            "onnx/embed_tokens_q4f16.onnx": (1065, "bab0a291678334b4463d0d5c6e65fbcc1169fea9c1ebcbb849176c4e1baaf8de"),
            "onnx/embed_tokens_q4f16.onnx_data": (440401920, "a0c032f7d6a902861b58b9f2432a6f9128ecc0c0e85bffed3089972ccc60f96b"),
        },
        "optional": {},
    },
    "ministral-3-8b": {
        "repo": "hyperquader/ministral-3-8b-instruct-genai-int4",
        "revision": "79446910a88a08c4b611cf6f48c12cac7aae48ff",
        "files": {
            "tokenizer_config.json": (21165, "a178ed03c7fd810d021b97045f65572ab6a43279d260510fbb09e13a7bd66da5"),
            "tokenizer.json": (17077420, "99cf274236c60277fcfad861a5a1007518687ad06ba8938760f50b55ffa0b1ef"),
            "cuda/model.onnx": (253259, "58838daee8b49a46ea7b5a687db6d22d7eb7ddea47b8e58dd75624d6068c4751"),
            "cuda/model.onnx.data": (5614665728, "888e4c678062ba1f2dcb9c833754f703fe3d06a8c808a338c960ed4d561cbae5"),
        },
        "optional": {},
    },
    "smollm3-3b": {
        "repo": "HuggingFaceTB/SmolLM3-3B-ONNX",
        "revision": "af50613703fb6f10ffcb21b27ad48edcb8334232",
        "files": {
            "tokenizer_config.json": (56256, "563f3b7982aad17432e2cb4f98db0fcaf6246f02ce46c2d41b6ec0aefe08544b"),
            "chat_template.jinja": (5575, "6f8839eb2d251bfcd822dd9381cab97607e488ba4d18fbbb40560bf506e2a9ec"),
            "tokenizer.json": (11574059, "900d675a509dc23ad4df99adc7e70f83bab6e981578e2263362744e1cfac4ca2"),
            "onnx/model_q4f16.onnx": (301534, "611ecc276b84aa6e01e073fbb4d9fc9aec59cd4584f3d6cf19d19d7f46d1df4c"),
            "onnx/model_q4f16.onnx_data": (2124320768, "de9041ea9b8a3f5ee102142dc9fb4684a111616bac4d9d9bac5f945e01a816e2"),
        },
        "optional": {},
    },
    "gemma-3n-e2b-it": {
        "repo": "onnx-community/gemma-3n-E2B-it-ONNX",
        "revision": "d3068b2ea2b9e9de85b33cb356121c4ca0510c0c",
        "files": {
            "tokenizer_config.json": (2644, "e5381f672b3b65a1420bc6b49b61f0b28a8b97d849217b6ecf2a2249ecb18e1f"),
            "chat_template.jinja": (1626, "ac03dcb3b09726f1e50ac55ae58fc8d5930eadc3b3a12cb04286dc1d82ac8001"),
            "tokenizer.json": (20366294, "44cb3d7d545cf895311e004d9a2b2ce823be5eb84c9aa31f73858b607c44c924"),
            "onnx/decoder_model_merged_q4.onnx": (1686685, "4fcb3a37937db577756270c504851e9366ffa738ace6c5ee7d345728aa8dcbd0"),
            "onnx/decoder_model_merged_q4.onnx_data": (1620499456, "297a9301058969f1e67e42546a48875b4250f58b10a28249ff08d76e0b5ead57"),
            "onnx/embed_tokens_q4.onnx": (3413, "54431e64a782ed74220a0af6b4184a6bf59e4b47e0b14ea4e1335810bc0f4f9b"),
            "onnx/embed_tokens_q4.onnx_data": (1634017280, "a94d38d81555f7143d1f2108433f866aa54cd4b574e3a527e3edacc34c475d70"),
        },
        "optional": {},
    },
    "gemma-4-e2b-it": {
        "repo": "onnx-community/gemma-4-E2B-it-ONNX",
        "revision": "7c6d3d1d4092253ea241428e88312ab34bfa9c26",
        "files": {
            "tokenizer_config.json": (19795, "e5d1a3b4bfd10a843cff2189b01ddd30c7f646d6b2998481eb93a9c6e9487a90"),
            "chat_template.jinja": (16317, "781d10940fbc44be40064b5d43a056fc486c84ceaa55538226368b57314132bf"),
            "tokenizer.json": (19439251, "47bd35616c7c782aaca6ccf48c75f3461d5877170984b8836b375107d0a9f566"),
            "onnx/decoder_model_merged_q4f16.onnx": (673231, "73c0f1fe04f9a3a048fb3319c0671b6cf0346bf33a3a8624c853bcffe01c24a4"),
            "onnx/decoder_model_merged_q4f16.onnx_data": (1519700992, "3b27245a7396cb7039a4e4118bd2a8aa35106bae381522edf7c4867b5f22bb10"),
            "onnx/embed_tokens_q4f16.onnx": (5621, "d7ca53f6a169471b5699b2f57ee4c7aa2c73732b0152f3909e64b71384444825"),
            "onnx/embed_tokens_q4f16.onnx_data": (1590689792, "024b199e6358ed42970f807686add5f9430d7e254ca7ce22fc9c83f015b9c517"),
        },
        "optional": {},
    },
    "gemma-4-e4b-it": {
        "repo": "onnx-community/gemma-4-E4B-it-ONNX",
        "revision": "d298100958fdf3c60581158fe201fb240634fb28",
        "files": {
            "tokenizer_config.json": (19795, "e5d1a3b4bfd10a843cff2189b01ddd30c7f646d6b2998481eb93a9c6e9487a90"),
            "chat_template.jinja": (16317, "781d10940fbc44be40064b5d43a056fc486c84ceaa55538226368b57314132bf"),
            "tokenizer.json": (19439251, "47bd35616c7c782aaca6ccf48c75f3461d5877170984b8836b375107d0a9f566"),
            "onnx/decoder_model_merged_q4f16.onnx": (850610, "43aa27452be3dd7fbb9524257dd66af957add748ddab20ea63ae71923e59aa08"),
            "onnx/decoder_model_merged_q4f16.onnx_data": (2074847232, "b6aa13eab3ecdf4721293e93c806c279ca0516956187f7aec63ee90ec7216e73"),
            "onnx/decoder_model_merged_q4f16.onnx_data_1": (812318720, "84e1c5f09ba88a5351959e4f73f62bce46f92dc19a7d7c82376ef36771c26a30"),
            "onnx/embed_tokens_q4f16.onnx": (5619, "aa48aa1806eda0ea42b79cd8eea355aebaf3b6ae3b04190bfee7ceef308603a4"),
            "onnx/embed_tokens_q4f16.onnx_data": (2017460224, "fd0f39c08f7e20a31145c2351a76a408b6c4ab60d15cc33f40e29cf30c0b2451"),
        },
        "optional": {},
    },
    "gpt-oss-20b": {
        "repo": "onnx-community/gpt-oss-20b-ONNX",
        "revision": "9684648d42b22f9602ec8b7b21e584dd865cf167",
        "files": {
            "tokenizer_config.json": (21688, "c8745c42d2a42b9ff67231478b220c3ed4867b5711323a4c05cf227bbe2f14f2"),
            "chat_template.jinja": (16330, "e381f0b98be3c3714676ec4102bc75a885d75b9002feb02fb6f25ad0e641bce3"),
            "tokenizer.json": (27868174, "0614fe83cadab421296e664e1f48f4261fa8fef6e03e63bb75c20f38e37d07d3"),
            "onnx/model_q4f16.onnx": (55192622, "5834cda2698332a02b1451a2935d10e7484b871bfa30c3a8b62cb72cc74e6a83"),
            "onnx/model_q4f16.onnx_data": (2070839296, "2fb5a8564a09f6521e51b06f29384439b1285ccf9166533fceb4ed6af206aa7a"),
            "onnx/model_q4f16.onnx_data_1": (2057011200, "0d3370b50d98413348135365c925b6af46346171693041c115d96f79c3f4fe21"),
            "onnx/model_q4f16.onnx_data_2": (1974067200, "ff26e086a3887371ac51a89aeac752cbe2dce20cec514a2190d54287d302b4d4"),
            "onnx/model_q4f16.onnx_data_3": (2057011200, "d4c49f1cb2f941bc5d457546c0b35707b518e0401d4dd9716238f45b7d6ef235"),
            "onnx/model_q4f16.onnx_data_4": (1974067200, "3619260fcfd26b367cd886f1d21a24332f111ded159ca388b7e0d8f1026b414e"),
            "onnx/model_q4f16.onnx_data_5": (2096824320, "4848013f1019367d427e5802c58b0adeb4f5cae31e1ff639fa39f9e70ef6660c"),
            "onnx/model_q4f16.onnx_data_6": (339033600, "74f352935e9af66748ad0afe803aee88f94735865ac3a6ff3f3380885fb1c18c"),
        },
        "optional": {},
    },
    # P3 models (2026-10-12)
    "qwen3.5-9b": {
        "repo": "amd/Qwen3.5-9B_gpuep_1.0_gpu_128",
        "revision": "bad679999a75e4883384cdac008972614cfb0d00",
        "files": {
            "text.onnx": (661516, "723a8630cfbbced734c56c92ccacf3fd3e1332ea9a0cebe7765179c59fb810f0"),
            "text.onnx.data": (4533018624, "a7f8bc812ed3d41fd9b17eb356794c0fc170f612aed29b29f0d71d36c244e35a"),
            "embedding.onnx": (3018, "645c888c0943066cb87a039436ca4620ee719072494467daa4261297160ee417"),
            "embedding.onnx.data": (2034237440, "a781bf4004f3b265d676800124628ddcda261ee90bc7c6d560f201f3eb31cbe2"),
            "tokenizer.json": (11071263, "6f57b7081304082fec953d5087f68107c261685f3d73f816233acffb088f8268"),
            "tokenizer_config.json": (1124, "eae0b680a277b7a210259360b8a1c2351160fb55e87f3dc38dfb32c59f29a1b2"),
            "chat_template.jinja": (7756, "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715"),
            "genai_config.json": (2874, "6c65a0d7a81a94bea17e3dd7c6f36f4aa8d915f455f0569af59af95ff3b52ba8"),
            "config.json": (2809, "4a627ba2d73c6c19a98e3241f5b69a7a25391f741802e2a5abcab276ba0251bc"),
        },
        "optional": {},
    },
    "qwen3.8-27b": {
        "repo": "amd/Qwen3.8-27B-fp16-ve-fp16-int4-k_quant-gs128-text-onnx",
        "revision": "3d00750aaa18e3e09e0b271a82e7762a4586718f",
        "files": {
            "text.onnx": (1413840, "44b712df872250c1c1614c3aa717788a103f3a814e0cfd24b22196d3b99509cd"),
            "text.onnx.data": (13394313216, "9944f22d543669858768ab365cd8b09af68117396cf7e030c10635468db8a0e3"),
            "embedding.onnx": (3018, "58f07f986c7a49693e85258df282134f29b5d158702ff334f228ecfc738dadb2"),
            "embedding.onnx.data": (2542796800, "41a8b66f4d3d229db720eb18f0e4567d721e784c3ef8f891274cfe1a362473bd"),
            "tokenizer.json": (11071263, "6f57b7081304082fec953d5087f68107c261685f3d73f816233acffb088f8268"),
            "tokenizer_config.json": (1155, "3248af4991a821494d21aa4edaec4a858d16f0839c24c852a570833f23f6f714"),
            "chat_template.jinja": (9127, "27dd380a2dd2cd6d7c9645609395c886aa8fcce4ac08c6d1c943c209ba5701bc"),
            "genai_config.json": (2797, "5c3698196aa4ab74a8e997ef2750217cdc6afdefadd26741023c9b48f8ba54e1"),
            "config.json": (3834, "bbfc9a2934e1cae485e88789748ae5d8ecfe2e0cb88b4e5ffdfab99f806b6aba"),
        },
        "optional": {},
    },
    "qwen3.6-35b-a3b": {
        "repo": "amd/Qwen3.6-35B-A3B_gpuep_1.0_gpu_128",
        "revision": "c2fdd8966cfde2da5fa780004fe6fad4a1125bb8",
        "files": {
            "text.onnx": (1018579, "1b26953b1fa577565c3ee423c2525b19ac91826e172717e7a836078998a6796d"),
            "text.onnx.data": (19280867328, "0a2e381b4a2ef5afcc08c8210268380eec160dc24a7b86086c2eaf854e1b3f58"),
            "embedding.onnx": (3161, "af5aadad8bd7188023fa4a45e93fb337bc285cef7a433f5be6ccd36dce7bf6de"),
            "embedding.onnx.data": (1017118720, "784757309ed420d85a31aa82b0d6ed3260e3e4acb85036b6fbc16bce52dc4210"),
            "tokenizer.json": (11071263, "6f57b7081304082fec953d5087f68107c261685f3d73f816233acffb088f8268"),
            "tokenizer_config.json": (1124, "eae0b680a277b7a210259360b8a1c2351160fb55e87f3dc38dfb32c59f29a1b2"),
            "chat_template.jinja": (7764, "e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259"),
            "genai_config.json": (2810, "3f3fd3854432c0921c69dc96e89ac07d16e3f5df8079e601158c77b37e505f7b"),
            "config.json": (3200, "dd81fe1c724a035987764b240b8ca5a638157bafb3e5ec657a8f0c2bc808ec1b"),
        },
        "optional": {},
    },
    "mistral-nemo-12b": {
        "repo": "nvidia/Mistral-Nemo-12B-Instruct-ONNX-INT4",
        "revision": "1d8d9911b82b3a42c940cfeaf2cde3077da037cf",
        "files": {
            "model.onnx": (545151, "219f817360446debb5e2b557f2c2e40dc6a5cfc2c9a132ede25b9278b1344372"),
            "model.onnx_data": (8573839360, "9ef8e2c8550824c282e318d9b074d4c93d054282c37d6493f03015ee825a2b36"),
            "tokenizer.json": (17078292, "b0240ce510f08e6c2041724e9043e33be9d251d1e4a4d94eb68cd47b954b61d2"),
            "tokenizer_config.json": (189327, "08ec78811fb6eef9aaae9a4e562691a55c1403a1b0acd7840dd63b952cc784d6"),
            "genai_config.json": (1662, "890b1f2cc3e3680ee9764adb0d29a124ffb6df153e3d816d98017dc24613b052"),
            "special_tokens_map.json": (437, "b779ce0b2be00f7cfd9d6188ed88e94700dafc1a7138fef8fca88f06aca11dac"),
        },
        "optional": {},
    },
    "granite-4.1-8b": {
        "repo": "LemOneLabs/Granite-4.1-8b-Onnx",
        "revision": "ffa38228d972bdd3ebfe3f847d4e9058a4ad2e20",
        "files": {
            "model.onnx": (952187, "29c2a8902948925c2811d4fd8150a364f5f4c1e09042d81c0c24037474200710"),
            "model.onnx.data": (16794714112, "e58c284851aa176d9bad933f47118da9f81abc0b5a7dc6f32639c5f1ae4bc994"),
            "tokenizer.json": (7153434, "24665f2815ee47aba521e4710bfc52f8ca045a552ec3f60925cce5e0edecb657"),
            "tokenizer_config.json": (401, "af849197b7d78d5bc3f29e0b594886476405e4fe3a28a33c1cc2292c00f5d0bc"),
            "chat_template.jinja": (6212, "735184a4c7cadc955b11f322926ac95a5961a106ae9028f85d984fc08f122073"),
            "genai_config.json": (1578, "c00000106831147e1835fe62dbb75a2caa26196f98f62508128254a40db0b5f8"),
            "config.json": (505, "fd22c7015dce2a1b7f0399a97f29a42ab27e55741bce5ab7479df1501a15703f"),
        },
        "optional": {},
    },
    "mistral-small-24b": {
        "repo": "elbruno/Mistral-Small-24B-Instruct-onnx",
        "revision": "1576c17c8ea524d2397ff2555d84b1929aa08128",
        "files": {
            "model.onnx": (299350, "58b3e3a8be005535545eb1e778d7af5aecaa7049133e11cee6c425a42c439fae"),
            "model.onnx.data": (17015898112, "32cd719b60589196c347a4dcba2735ec2656f2429bdc459b55890409a25e0226"),
            "tokenizer.json": (17078037, "b76085f9923309d873994d444989f7eb6ec074b06f25b58f1e8d7b7741070949"),
            "tokenizer_config.json": (207047, "67ba10795c8b8013518477d7a0e887f7691ad87faccca0b3dbaad944cdcb8663"),
            "chat_template.jinja": (1621, "3e28eac6e69f1bd645b8d069cf80e8129b90418bbc05db8fecb69f588ea494c3"),
            "genai_config.json": (1511, "19e422617de51f26931f3aeb93366c8d70591775ca99fd94e36fd0ceb5da6c37"),
            "special_tokens_map.json": (22336, "96984b1207b98da02c8b34c2e680b715ecdc493d99302c1d198445c8c2b35e7b"),
        },
        "optional": {},
    },
    "gemma-4-26b-a4b-it": {
        "repo": "amd/Gemma-4-26B-A4B-it_gpuep_1.0_gpu_128",
        "revision": "43aff7f47ce099889f5f8046395b03e5ec5b12d1",
        "files": {
            "decoder/model.onnx": (1110985, "cd059b615afcdd127d89cd571eddf498383b022b3cbe23e7d36c5a77747cb46b"),
            "decoder/model.onnx.data": (14638448640, "d30ad84cf42935f063c7154f9597e61193a20c743c492f60bb817da39d0cd49c"),
            "embedding/model.onnx": (5264, "84374292ae060fffaab24eca56f537826c6ea8e52e87806802604c4a28b62f55"),
            "embedding/model.onnx.data": (1476460544, "32774b7fd689695443144352438e5701d624a0aebdcd007f1c4ab8ffef2eb39b"),
            "google/tokenizer.json": (32169626, "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f"),
            "google/tokenizer_config.json": (3082, "9f4fec4b1dc6ecddf8f4a92e9caea5971c0e67d81309f3f9066a2bee8c362633"),
            "google/chat_template.jinja": (18683, "ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4"),
        },
        "optional": {},
    },
    "gemma-4-12b-it": {
        "repo": "RASMUS/gemma-4-12B-it-ONNX",
        "revision": "b1c8b3e73e0440da277f1a6fec7128ab211b9986",
        "files": {
            "onnx/decoder_model_merged_q4f16.onnx": (16013678, "62dea895f6b4d9a2fde710201909b8c7d91bc36fd003c86d91400a6f1717cef8"),
            "onnx/decoder_model_merged_q4f16.onnx_data": (2036441088, "40357ee6115efa33ca0fb39a80c4ce36ebc751e7e4feb3d27fe702a4b5fd319d"),
            "onnx/decoder_model_merged_q4f16.onnx_data_1": (2017566720, "3f3722fb60eb3df6425df4ac0436ef2d73214047b52ab47a49c7d287d82454b5"),
            "onnx/decoder_model_merged_q4f16.onnx_data_2": (2013880320, "6f280b1e0d3844c6372e4a503ee71a587a0d67295a966f693005cd3698aaa9d9"),
            "onnx/decoder_model_merged_q4f16.onnx_data_3": (632586240, "6eca42076fdc99b1635b50c0fdca8b4b4ddebffe87665dbe71c770406a328487"),
            "onnx/embed_tokens_fp16.onnx": (4025, "87ac505f4b5b850018fcdb2dcef8107c168f32da7d9278540bef5551940aac16"),
            "onnx/embed_tokens_fp16.onnx_data": (2013265920, "fb063e8fd1274b19d4f3d823b7da0c575eb57a32cd741df965b52f211ddb6dcf"),
            "tokenizer.json": (32169626, "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f"),
            "tokenizer_config.json": (22222, "589767afaacec6504fef7cc584b8ec703ed7d92b19c86fc2df459d476fc78e8c"),
            "chat_template.jinja": (18683, "ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4"),
            "config.json": (3315, "f8755967354402c389e3197b6a46a32e4edec58a68ba12ca4b867a420c4b69b1"),
        },
        "optional": {},
    },
    # end P3 models (2026-10-12)
    # own exports (2026-10-13)
    "qwen2.5-7b-instruct": {
        "repo": "danyalmalik/Qwen2.5-7B-Instruct-ONNX-WebGPU",
        "revision": "7131663cefd8305fa744eb8ba692b34703c7a9d1",
        "files": {
            "tokenizer.json": (11421892, "3fd169731d2cbde95e10bf356d66d5997fd885dd8dbb6fb4684da3f23b2585d8"),
            "tokenizer_config.json": (692, "4d8ad3fe9729e37338fae50bac94ff03cb915071eb886b937077114e874dfa06"),
            "chat_template.jinja": (2507, "cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f"),
        },
        "optional": {},
    },
    "qwen2.5-coder-7b-instruct": {
        "repo": "danyalmalik/Qwen2.5-Coder-7B-Instruct-ONNX-WebGPU",
        "revision": "2153ac803393d1d4875f820036e6c4e5707bb584",
        "files": {
            "tokenizer.json": (11421892, "3fd169731d2cbde95e10bf356d66d5997fd885dd8dbb6fb4684da3f23b2585d8"),
            "tokenizer_config.json": (692, "4d8ad3fe9729e37338fae50bac94ff03cb915071eb886b937077114e874dfa06"),
            "chat_template.jinja": (2507, "cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f"),
        },
        "optional": {},
    },
    "deepseek-r1-distill-qwen-7b": {
        "repo": "danyalmalik/DeepSeek-R1-Distill-Qwen-7B-ONNX-WebGPU",
        "revision": "a4940c386a2f5e412086038704528e4d8ecc7a3e",
        "files": {
            "tokenizer.json": (11421907, "f624f8136bc57382156e7595e8ea404a66fbce12e3181b933ec2459bda3a71be"),
            "tokenizer_config.json": (421, "3f314d1e3a39940a81a731a5770d41a02e07cff9c62002f7a39e78ab3c63869c"),
            "chat_template.jinja": (2246, "56a1447ad31926fdc21fb07e56e5642bd9c850c4f52d8c8af7bbe5f079a84f5f"),
        },
        "optional": {},
    },
    "deepseek-r1-0528-qwen3-8b": {
        "repo": "danyalmalik/DeepSeek-R1-0528-Qwen3-8B-ONNX-WebGPU",
        "revision": "502fcaec49e964486759bc9279be8b66f8259f6e",
        "files": {
            "tokenizer.json": (7032822, "47d3a3ffb1ff38eb6a22f8a2d207306bfa3ea2499ff70df4baeaa559bf7453c7"),
            "tokenizer_config.json": (3958, "78872f2c1f0a5fb79d5872dbacb4c83991ef416315a25f34659e6a7c34108027"),
        },
        "optional": {},
    },
    "glm-4.7-flash": {
        "repo": "danyalmalik/GLM-4.7-Flash-ONNX-WebGPU",
        "revision": "8a99d07a418f445e889dede77ccdf60883063862",
        "files": {
            "tokenizer.json": (20217442, "19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d"),
            "tokenizer_config.json": (7226, "31a173e2797ddc8b72ac996803513e627fc28d7aad02cfcce321a431d865c86d"),
            "chat_template.jinja": (3120, "d63ad536c3c81880043e22ec7fd08db42b4d8fb7c89c7138bc562bfa25281375"),
        },
        "optional": {},
    },
    "gemma-3-12b-it": {
        "repo": "danyalmalik/gemma-3-12b-it-ONNX-WebGPU",
        "revision": "e7e1cd11468a8a03ec20d702c4940c48262d9809",
        "files": {
            "tokenizer.json": (33384568, "4667f2089529e8e7657cfb6d1c19910ae71ff5f28aa7ab2ff2763330affad795"),
            "tokenizer_config.json": (1158492, "8925eabb556c1897ca1bed405f0612e735a56afcd4d77e76eb90fd771e706f9c"),
            "chat_template.jinja": (1532, "7de1c58e208eda46e9c7f86397df37ec49883aeece39fb961e0a6b24088dd3c4"),
        },
        "optional": {},
    },
    "nemotron-3-nano-30b-a3b": {
        "repo": "danyalmalik/NVIDIA-Nemotron-3-Nano-30B-A3B-ONNX-WebGPU",
        "revision": "4fd6c9487b84b6b78607c28d8dbeef059966e548",
        "files": {
            "tokenizer.json": (17077485, "c6021eb6847e682f89aa52d5eb6e8c7d902a23acfc8137e25211cf84828f1592"),
            "tokenizer_config.json": (188049, "3d568e506d0905285ae90a3fdd1482be7b6d0bf0b8ca9514d75dad4257b0827a"),
            "chat_template.jinja": (10504, "ab7813c3abdd9cb655905a410728b26c7884eca45ddfab8d9f931553485a7862"),
        },
        "optional": {},
    },
    # end own exports (2026-10-13)
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
