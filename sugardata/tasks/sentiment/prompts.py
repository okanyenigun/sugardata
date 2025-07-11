from ...components.translator import Translator


def get_dimension_prompt(language: str) -> str:

    DIMENSION_PROMPTS = {
        "en": """
                You are an expert in ontology design, knowledge representation, and concept hierarchy analysis.
                You will be given a concept and your task is to generate dimensions, sub fields, subcategories, and related concepts for that concept.
                You must generate at least 10 dimensions, ensuring a mix of hierarchical subcategories and adjacent concepts.
                
                ############
                Concept: {concept}

                Format Instructions:
                {format_instructions}
            """,
        "tr":"""
                Ontoloji tasarımı, bilgi temsili ve kavram hiyerarşisi analizi konusunda uzman birisiniz.
                Size bir kavram verilecek ve göreviniz bu kavram için boyutlar, alt alanlar, alt kategoriler ve ilgili kavramlar üretmektir.
                En az 10 boyut üretmelisiniz, hiyerarşik alt kategoriler ve ilgili kavramlar arasında bir karışım sağlamak için.
                Alt kategoriler (kavramın altında yer alanlar) ve ilgili kavramlar (yaratıcı bağlantılar kuranlar) arasında net bir ayrım yapmalısınız.

                ############
                Kavram: {concept}
                Format Talimatları:
                {format_instructions}
            """
    }

    text = DIMENSION_PROMPTS.get(language)

    if not text:
        core_text = DIMENSION_PROMPTS["en"].split("############")[0]
        translated_core_text = Translator.translate(core_text, target_language=language, source_language="en", vendor="deep-translator")
        text = translated_core_text + DIMENSION_PROMPTS["en"].split("############")[1]

    return text


def get_aspect_prompt(language: str) -> str:
    ASPECT_PROMPTS = {
        "en": """
                You are an expert in ontology design, knowledge representation, and concept hierarchy analysis.
                You will be given a concept and its a specific dimension (or subcategory) and your task is to generate aspects for that concept.

                ############
                Index: {index}
                Concept: {concept}
                Dimension: {dimension}

                Format Instructions:
                {format_instructions}
            """,
        "tr": """
                Ontoloji tasarımı, bilgi temsili ve kavram hiyerarşisi analizi konusunda uzman birisiniz.
                Size bir kavram ve onun belirli bir boyutu (veya alt kategorisi) verilecek ve göreviniz bu kavram için yönleri üretmektir.

                ############
                İndeks: {index}
                Kavram: {concept}
                Boyut: {dimension}

                Format Talimatları:
                {format_instructions}
            """
    }

    text = ASPECT_PROMPTS.get(language)

    if not text:
        core_text = ASPECT_PROMPTS["en"].split("############")[0]
        translated_core_text = Translator.translate(core_text, target_language=language, source_language="en", vendor="deep-translator")
        text = translated_core_text + ASPECT_PROMPTS["en"].split("############")[1]
    return text


def get_sentence_prompt(language: str) -> str:
    SENTENCE_PROMPTS = {
        "en": """
                You are an advanced creative writing engine specialised in sentiment-rich text generation.  
                Your goal is to craft ONE vivid, coherent snippet that unmistakably expresses the required **sentiment toward the given ASPECT** of a CONCEPT, while embodying every stylistic parameter supplied.

                ────────────────────────────────────────────────────────
                ❶ THINK & PLAN (hidden to user)  
                • Briefly list (max 40 words) how each parameter below will appear in the text (vocabulary, tone, devices, structure).  
                • Ensure the plan contains at least three concrete lexical or stylistic choices that differentiate it from typical prose for this task.  
                • End the plan with `### WRITE` on its own line.

                ❷ WRITE (visible to user)  
                • Produce the final snippet **after** the `### WRITE` marker only.  
                • Length target: 1– given sentence_length} sentences (±10 %).  
                • **Do NOT copy phrases** used in earlier calls within the same session; employ fresh metaphors, imagery, and syntax.  
                • Include at least two rhetorical or figurative devices (e.g., alliteration, metaphor, antithesis) suited to the chosen *writing_style* and *medium*.  
                • Prioritise, in order:  
                    1. **Medium** – its conventions should shape diction and structure.  
                    2. **Aspect-Sentiment** link – the feeling must be unambiguous.  
                    3. Other parameters (persona, intention, audience, register…).  
                • Keep language natural; avoid boilerplate sentiment clichés (“very good”, “extremely bad”).  
                • Reference the *aspect* explicitly at least once; reference the *concept* implicitly or explicitly.

                ############
                ────────────────────────────────────────────────────────
                INPUT PARAMETERS  
                Index: {index}  
                Concept: {concept}  
                Aspect: {aspect}  
                Writing Style: {writing_style}  
                Medium: {medium}  
                Persona: {persona}  
                Intention: {intention}  
                Sentence Length: {sentence_length}

                OUTPUT FORMAT  
                Dimension → Aspect → Sentiment  
                {format_instructions}
                """,
        "tr": """
                Gelişmiş bir duygu zengin metin üretimine özel bir yaratıcı yazma motorusunuz.
                Amacınız, verilen KAVRAM'ın belirli bir YÖNÜ'ne yönelik DUYGUYU açıkça ifade eden BİR canlı, tutarlı parça yazmaktır,
                aynı zamanda sağlanan her stil parametresini de içermektir.
                ────────────────────────────────────────────────────────
                ❶ DÜŞÜN & PLANLA (kullanıcıya gizli)
                • Aşağıdaki her parametrenin metinde nasıl görüneceğini (maksimum 40 kelime) kısaca listeleyin (söz dağarcığı, ton, araçlar, yapı).
                • Planın, bu görev için tipik bir anlatımdan ayıran en az üç somut sözlük veya stil seçeneği içermesini sağlayın.
                • Planı `### YAZ` ile bitirin kendi satırında.
                ❷ YAZ (kullanıcıya görünür)
                • Son parçayı `### YAZ` işaretinden SONRA üretin.
                • Uzunluk hedefi: 1–{sentence_length} cümle (±%10).
                • Aynı oturum içinde daha önceki çağrılarda kullanılan ifadeleri KOPYALAMAYIN; taze metaforlar, imgeler ve sözdizimi kullanın.
                • Seçilen yazım stili ve ortam için uygun en az iki retorik veya figüratif araç (örneğin, aliterasyon, metafor, antitez) ekleyin.
                • Öncelik sırası:
                    1. Ortam – söz dağarcığı ve yapıyı şekillendirmelidir.
                    2. Yön-Duygu bağlantısı – duygu açık olmalıdır.
                    3. Diğer parametreler (persona, niyet, izleyici, kayıt…).
                • Dili doğal tutun; şablon duygu klişelerini (örneğin, "çok iyi", "son derece kötü") kullanmaktan kaçının.
                • YÖN'ü en az bir kez açıkça, KAVRAM'ı ise dolaylı veya doğrudan referans edin.
                
                ############
                ────────────────────────────────────────────────────────
                GİRİŞ PARAMETRELERİ
                İndeks: {index}
                Kavram: {concept}
                Yön: {aspect}
                Yazım Stili: {writing_style}
                Ortam: {medium}
                Persona: {persona}
                Niyet: {intention}
                Cümle Uzunluğu: {sentence_length}
                ÇIKTI FORMAT
                Boyut → Yön → Duygu
                {format_instructions}
                """
    }

    text = SENTENCE_PROMPTS.get(language)

    if not text:
        core_text = SENTENCE_PROMPTS["en"].split("############")[0]
        translated_core_text = Translator.translate(core_text, target_language=language, source_language="en", vendor="deep-translator")
        text = translated_core_text + SENTENCE_PROMPTS["en"].split("############")[1]   
    return text


def get_augment_sentence_prompt(language: str) -> str:
    AUGMENT_SENTENCE_PROMPTS = {
        "en": """
                You are an advanced creative writing engine specialised in sentiment-rich text generation.  
                Your goal is to craft ONE vivid, coherent snippet that unmistakably expresses the required **sentiment toward the given ASPECT** of a CONCEPT, while embodying every stylistic parameter supplied.

                ────────────────────────────────────────────────────────
                ❶ THINK & PLAN (hidden to user)  
                • Briefly list (max 40 words) how each parameter below will appear in the text (vocabulary, tone, devices, structure).  
                • Ensure the plan contains at least three concrete lexical or stylistic choices that differentiate it from typical prose for this task.  
                • End the plan with `### WRITE` on its own line.

                ❷ WRITE (visible to user)  
                • Produce the final snippet **after** the `### WRITE` marker only.  
                • Length target: 1– given Sentence Length sentences (±10 %).  
                • **Do NOT copy phrases** used in earlier calls within the same session; employ fresh metaphors, imagery, and syntax.  
                • Include at least two rhetorical or figurative devices (e.g., alliteration, metaphor, antithesis) suited to the chosen *writing_style* and *medium*.  
                • Prioritise, in order:  
                    1. **Medium** – its conventions should shape diction and structure.  
                    2. **Aspect-Sentiment** link – the feeling must be unambiguous.  
                    3. Other parameters (persona, intention, audience, register…).  
                • Keep language natural; avoid boilerplate sentiment clichés (“very good”, “extremely bad”).  
                • Reference the *aspect* explicitly at least once; reference the *concept* implicitly or explicitly.
                You should never copy the given text, but you can use it as a reference to understand the context.
                
                ############
                ────────────────────────────────────────────────────────
                INPUT PARAMETERS  
                Index: {index}  
                Concept: {concept}  
                Aspect: {aspect}  
                Writing Style: {writing_style}  
                Medium: {medium}  
                Persona: {persona}  
                Intention: {intention}  
                Sentence Length: {sentence_length}
                Given Text: {given_text}

                OUTPUT FORMAT  
                Dimension → Aspect → Sentiment  
                {format_instructions}
                """,
        "tr": """
                Gelişmiş bir duygu zengin metin üretimine özel bir yaratıcı yazma motorusunuz.
                Amacınız, verilen KAVRAM'ın belirli bir YÖNÜ'ne yönelik DUYGUYU açıkça ifade eden BİR canlı, tutarlı parça yazmaktır,
                aynı zamanda sağlanan her stil parametresini de içermektir.
                ────────────────────────────────────────────────────────
                ❶ DÜŞÜN & PLANLA (kullanıcıya gizli)
                • Aşağıdaki her parametrenin metinde nasıl görüneceğini (maksimum 40 kelime) kısaca listeleyin (söz dağarcığı, ton, araçlar, yapı).
                • Planın, bu görev için tipik bir anlatımdan ayıran en az üç somut sözlük veya stil seçeneği içermesini sağlayın.
                • Planı `### YAZ` ile bitirin kendi satırında.
                ❷ YAZ (kullanıcıya görünür)
                • Son parçayı `### YAZ` işaretinden SONRA üretin.
                • Uzunluk hedefi: 1–{sentence_length} cümle (±%10).
                • Aynı oturum içinde daha önceki çağrılarda kullanılan ifadeleri KOPYALAMAYIN; taze metaforlar, imgeler ve sözdizimi kullanın.
                • Seçilen yazım stili ve ortam için uygun en az iki retorik veya figüratif araç (örneğin, aliterasyon, metafor, antitez) ekleyin.
                • Öncelik sırası:
                    1. Ortam – söz dağarcığı ve yapıyı şekillendirmelidir.
                    2. Yön-Duygu bağlantısı – duygu açık olmalıdır.
                    3. Diğer parametreler (persona, niyet, izleyici, kayıt…).
                • Dili doğal tutun; şablon duygu klişelerini (örneğin, "çok iyi", "son derece kötü") kullanmaktan kaçının.
                • YÖN'ü en az bir kez açıkça, KAVRAM'ı ise dolaylı veya doğrudan referans edin.
                Verilen metni ASLA KOPYALAMAYIN, ancak bağlamı anlamak için referans olarak kullanabilirsiniz.

                ############
                ────────────────────────────────────────────────────────
                GİRİŞ PARAMETRELERİ
                İndeks: {index}
                Kavram: {concept}
                Yön: {aspect}
                Yazım Stili: {writing_style}
                Ortam: {medium}
                Persona: {persona}
                Niyet: {intention}
                Cümle Uzunluğu: {sentence_length}
                Verilen Metin: {given_text}

                ÇIKTI FORMAT
                Boyut → Yön → Duygu
                {format_instructions}
                """
    }

    text = AUGMENT_SENTENCE_PROMPTS.get(language)

    if not text:
        core_text = AUGMENT_SENTENCE_PROMPTS["en"].split("############")[0]
        translated_core_text = Translator.translate(core_text, target_language=language, source_language="en", vendor="deep-translator")
        text = translated_core_text + AUGMENT_SENTENCE_PROMPTS["en"].split("############")[1]   
    return text


def get_structure_prompt(language: str) -> str:
    STRUCTURE_PROMPTS = {
        "en": """
                You are an expert on extracting structured information from text.
                Your task is to analyze the provided text and extract structured information from it.
                The attributes you need to extract are:
                1. Concept: The main concept for which the sentiment is being analyzed.
                2. Aspects: The aspects related to the concept, which can have multiple derivatives. Only the aspects that the text is about should be extracted. Don't create hypothetical aspects. If not clear, just say "general".
                3. Writing Style: The writing style used in the generated text.
                4. Medium: The medium for which the text is generated.
                5. Persona: The persona of the writer.
                6. Intention: The intention behind the text.
                7. Sentence Length: The length of the sentences used in the text.
                8. Given Text: The text provided as input for structure extraction.

                Don't hypothetically create any information, just extract the information from the text.

                #############
                Text: {text}

                Format Instructions:
                {format_instructions}
            """,
        "tr": """
                Metinden yapılandırılmış bilgi çıkarma konusunda uzman birisiniz.
                Göreviniz, verilen metni analiz etmek ve yapılandırılmış bilgi çıkarmaktır.
                Çıkarmanız gereken öznitelikler şunlardır:
                1. Kavram: Duygunun analiz edildiği ana kavram.
                2. Yönler: Kavramla ilgili yönler, her yönün birden fazla türevi olabilir. Metnin ilgili olduğu yönleri çıkarmalısınız. Hipotetik yönler oluşturmayın. Net değilse, sadece "genel" olarak belirtin.
                3. Yazım Stili: Üretilen metinde kullanılan yazım stili.
                4. Ortam: Metnin hedeflendiği ortam.
                5. Persona: Yazarın karakteri.
                6. Niyet: Metnin arkasındaki niyet.
                7. Cümle Uzunluğu: Metinde kullanılan cümlelerin uzunluğu.
                8. Verilen Metin: Yapılandırma çıkarımı için verilen metin.

                Hipotetik olarak herhangi bir bilgi oluşturmayın, sadece metinden bilgiyi çıkarın.

                #############
                Metin: {text}

                Format Talimatları:
                {format_instructions}
            """
    }
    text = STRUCTURE_PROMPTS.get(language)

    if not text:
        core_text = STRUCTURE_PROMPTS["en"].split("#############")[0]
        translated_core_text = Translator.translate(core_text, target_language=language, source_language="en", vendor="deep-translator")
        text = translated_core_text + STRUCTURE_PROMPTS["en"].split("#############")[1]     
    return text

