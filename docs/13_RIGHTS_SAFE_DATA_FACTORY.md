# Rights-safe Korean embedding data factory v1

> 검토 기준일: 2026-07-11. 이 문서는 데이터 엔지니어링용 보수적 정책이며 법률 자문이나 권리 보증이 아니다. 권리 또는 이용조건이 불명확하면 수집 성공 여부와 무관하게 `default-deny`한다.

## 결론

공개 재배포 가능한 100만 행을 목표로 하는 실현 가능한 설계다. 실제 공개 승인은 아래 release gate를 모두 통과한 행에만 내려진다. “인터넷에 공개됨”, “공공기관 사이트”, “Open API”, “무료 열람”을 라이선스로 간주하면 안 된다. v1은 다음 다섯 소스군만 release 후보로 삼고, 문서마다 권리 근거와 revision을 보존한다.

| 소스군 | 목표 행 | 초기 판정 | 공개 데이터셋 조건 |
|---|---:|---|---|
| 공공누리 제0·1유형 어문 저작물 | 400,000 | conditional go | 개별 저작물에 0/1유형이 명시되고 제3자 권리·개인정보가 없어야 함 |
| 국가법령정보의 법령·조약·행정규칙 원문 | 250,000 | conditional go | 승인받은 공식 API만 사용하고 원문과 포털 편집·해설을 구분 |
| 한국어 위키백과 | 180,000 | conditional go, 별도 shard | CC BY-SA 의무와 문서별 revision/attribution을 계승 |
| PMC OA의 CC0/CC BY 본문 | 120,000 | conditional go | 논문별 기계판독 라이선스 확인, NC/ND/Other/누락 제외 |
| 미국 연방정부 직접 작성 자료 | 50,000 | legal review | 직원 직무저작인지와 제3자 삽입물, 역외 권리, 기관별 조건 확인 |
| 합계 | 1,000,000 | release gate 통과분만 | 부족분은 공공누리 +30K, 법령 +20K로 대체 |

OpenDART 원문은 v1의 100만 행에 포함하지 않는다. AIHub와 공개 벤치마크도 공개 데이터 팩토리에는 포함하지 않는다. 이 rights-safe 트랙은 별도 non-commercial 성능 트랙의 실험을 막지 않는다. 단, 두 트랙의 데이터와 산출물은 manifest, 저장소, 모델 카드에서 명확히 분리해야 한다.

정책의 기계 판독본은 [`configs/data_sources_v1.json`](../configs/data_sources_v1.json)이다.

## 1. 권리 판단 모델

하나의 `license` 문자열로 승인하지 않는다. 각 문서는 다음 네 층을 모두 통과해야 한다.

1. **소재 권리**: 법령 원문, 정부 직원의 직무저작, 논문 본문처럼 실제 텍스트의 권리 상태를 확인한다.
2. **편집·데이터베이스 권리**: 개별 소재가 자유여도 대량 선택·배열·반복 추출에는 별도 권리가 있을 수 있다. 대한민국 저작권법 제93조는 데이터베이스 전부 또는 상당 부분과 체계적인 반복 이용을 별도로 다룬다.
3. **전달 수단 조건**: API 승인, 호출 제한, 허용 목적, attribution 등 서비스 조건을 확인한다.
4. **비저작권 권리**: 개인정보, 초상·퍼블리시티, 상표, 특허, 영업비밀, 계약, 연구윤리 및 안전 문제를 별도로 검사한다.

대한민국 저작권법 제7조는 법률·조약·명령·조례·규칙, 일정한 고시·공고·훈령, 법원의 판결·결정 등을 보호받지 못하는 저작물로 열거한다. 이것은 “정부 사이트의 모든 글”이나 “모든 사실”이 자동으로 public domain이라는 뜻이 아니다. 원문과 민간 작성 해설, 기관의 창작성 있는 편집, DB 전체를 구분한다.

| 대상 | v1 판단 | 이유 |
|---|---|---|
| 법령·조약·일정한 행정규칙의 공식 원문 | 조건부 사용 | 제7조 근거가 강하지만 공식 API 승인과 원문 revision을 지켜야 함 |
| 개별 수치·날짜·식별자 같은 비창작적 사실 | 원문과 분리해 검토 | 사실이라는 이유만으로 그 DB의 체계적 대량 추출·재배포까지 허용되는 것은 아님 |
| 기관·기업·전문가가 작성한 설명과 요약 | 라이선스 없으면 제외 | 창작성 있는 표현일 수 있음 |
| 포털의 선택·배열·분류·검색 DB | API/DB 조건 별도 검토 | 소재의 상태와 편집저작물·데이터베이스제작자 권리는 별개 |

공식 근거:

- [저작권법 제7조 — 보호받지 못하는 저작물](https://www.law.go.kr/LSW/lsSideInfoP.do?docCls=jo&joBrNo=00&joNo=0007&lsiSeq=283335&urlMode=lsScJoRltInfoR)
- [저작권법 제2조 — 편집물·데이터베이스 정의](https://www.law.go.kr/LSW/lsSideInfoP.do?docCls=jo&joBrNo=00&joNo=0002&lsiSeq=283335&urlMode=lsScJoRltInfoR)
- [저작권법 제93조 — 데이터베이스제작자의 권리](https://www.law.go.kr/lsLinkCommonInfo.do?chrClsCd=010202&lsJoLnkSeq=1029423451)
- [공공데이터포털 이용정책 — 제3자 권리는 별도 허락 필요](https://www.data.go.kr/ugs/selectPortalPolicyView.do)

## 2. 소스별 판정과 수집 규칙

### 2.1 공공누리 제0·1유형: 주력 한국어 소스

[공공누리 유형안내](https://www.kogl.or.kr/info/license.do)는 현재 다음을 명시한다.

- 제0유형: 출처표시 조건 없이 상업·비상업 이용 및 2차적 저작물 작성 가능
- 제1유형: 출처표시를 조건으로 상업·비상업 이용 및 2차적 저작물 작성 가능
- 제2유형: 비영리 조건
- 제3유형: 변경 금지
- 제4유형: 비영리 + 변경 금지
- AI유형: AI 학습 목적의 이용은 가능하지만, 그 저작물로 만든 AI 학습용 데이터의 재판매를 금지하는 별도 조건이 있음

따라서 공개·상업 이용 가능한 학습 데이터 v1은 **개별 항목에 붙은 제0 또는 제1유형만** 받는다. AI유형만 붙은 항목, 제2~4유형, 유형 누락, 게시판 단위 표기와 첨부물의 관계가 불명확한 경우는 제외한다. AI유형과 제0/1유형이 함께 있어도 공개 dataset은 0/1유형 조건을 근거로 처리하고 그 사실을 기록한다.

수집점:

- 검색·원문 연결: [공공누리 원문 공공저작물](https://www.kogl.or.kr/recommend/recommendList.do)
- 유형 및 조건: [공공누리 유형안내](https://www.kogl.or.kr/info/license.do)
- 기관 적용 지침: [공공누리 온라인 적용방법](https://www.kogl.or.kr/info/publicGuide.do)
- 메타데이터 후보: [공공데이터포털의 한국문화정보원 공공저작물 자료](https://www.data.go.kr/data/15088592/fileData.do)

필수 검사:

- 항목 상세 페이지와 실제 첨부파일에 적용되는 유형을 각각 저장한다.
- `publisher`, 제목, 작성자, 작성연도, 원문 URL, 항목 URL, 이용유형, 이용유형 확인 URL, 수집시각을 저장한다.
- 제1유형은 배포하는 행과 데이터셋 attribution manifest 양쪽에 기관명·제목·연도·원문 링크를 남긴다.
- 저작인격권 훼손 우려가 있는 문맥 왜곡, 수치 변경, 기관 후원으로 오인시키는 표현을 금지한다.
- 제3자 저작권, 사진·도표, 상표, 개인정보가 섞인 부분은 텍스트 전체가 0/1유형처럼 보여도 제거한다.

### 2.2 국가법령정보: 원문은 유력하지만 전달 계층을 지킨다

[국가법령정보 공동활용 이용안내](https://open.law.go.kr/LSO/information/guide.do)는 법령정보를 영리 목적을 포함해 자유롭게 활용할 수 있다고 설명하면서도, 모든 데이터 활용은 신청·승인 후 가능하고 일부 API는 상업 이용 여부 등에 따라 제한될 수 있다고 고지한다.

공식 수집점:

- 전체 안내와 저작권 정책: <https://open.law.go.kr/LSO/information/guide.do>
- API 목록·본문 호출 설명: <https://open.law.go.kr/LSO/openApi/openApiManual.do>
- API별 가이드 191종: <https://open.law.go.kr/LSO/openApi/guideList.do>
- 목록 예시: `https://www.law.go.kr/DRF/lawSearch.do?OC={APPROVED_OC}&target=eflaw&type=JSON&display=100&page={PAGE}`
- 본문 예시: `https://www.law.go.kr/DRF/lawService.do?OC={APPROVED_OC}&target=eflaw&type=JSON&MST={MST}`

v1은 법령·조약·행정규칙의 공식 원문과 구조 메타데이터만 쓴다. 생활법령 해설, 세계법제 번역, 대학·공공기관 규정, 판례의 제3자 인용문, 별표의 이미지·서식은 원문과 권리 근거가 달라 자동 승인하지 않는다. 판결 원문 자체가 제7조에 포함되더라도 개인정보와 민감정보 때문에 v1에서는 보류한다.

API 인증값은 환경변수에서 주입하고 데이터·manifest·로그에 저장하지 않는다. 승인된 이용 목적과 실제 공개 데이터셋 용도가 달라지면 재승인 또는 서면 확인 전까지 release를 중단한다.

### 2.3 한국어 위키백과: 사용 가능하지만 share-alike 격리

공식 dump:

- 현재 dump 인덱스: <https://dumps.wikimedia.org/kowiki/latest/>
- 본문 dump: `https://dumps.wikimedia.org/kowiki/{YYYYMMDD}/kowiki-{YYYYMMDD}-pages-articles-multistream.xml.bz2`
- checksum: `https://dumps.wikimedia.org/kowiki/{YYYYMMDD}/kowiki-{YYYYMMDD}-sha1sums.txt`

[Wikimedia Terms of Use §7](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use#7._Licensing_of_Content)는 기여 텍스트를 CC BY-SA 4.0과 GFDL 조건으로 제공하고 attribution 방법을 설명한다. v1은 재현 가능한 날짜 dump로 `latest`를 해석한 뒤 날짜, 파일명, 크기, SHA-1/SHA-256을 고정한다. `latest` 문자열 자체를 revision으로 저장하지 않는다.

각 행은 `page_id`, `revision_id`, 제목, oldid URL, 문서 history URL, dump 날짜를 보존한다. 이미지·미디어·인용 블록·외부 표는 제외한다. 생성 query와 evidence pair가 adaptation인지에 대한 불확실성을 줄이기 위해 Wikipedia 파생 행은 다른 라이선스 자료와 섞어 단일 라이선스로 재배포하지 않고 `wikipedia_cc_by_sa` shard와 attribution manifest로 분리한다. 공개 전 share-alike 적용 범위는 별도 검토한다.

중요하게도 MIRACL과 KorQuAD/SQuADKor가 Wikipedia에서 만들어졌다. 현재 dump를 사용해도 benchmark 문서와 동일·유사한 구간은 query 생성 **전에** 제거한다.

### 2.4 PMC Open Access: “PMC에 있음”이 허락은 아니다

[PMC Open Access Subset 안내](https://pmc.ncbi.nlm.nih.gov/tools/openftlist/)는 PMC 전체가 재사용 가능한 것이 아니며, OA subset 안에서도 논문별 라이선스가 다르다고 명시한다. [PMC Copyright Notice](https://pmc.ncbi.nlm.nih.gov/about/copyright/)는 명시적 조건이 없으면 통상 저작권 보호를 가정하고, 허용된 자동 수집 경로만 사용하라고 한다.

허용 수집점:

- OA subset 정책: <https://pmc.ncbi.nlm.nih.gov/tools/openftlist/>
- OA API: <https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi>
- 논문 조회 예시: `https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id={PMCID}`
- FTP OA package: `https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_package/`

v1 whitelist는 `CC0`와 기계판독 가능한 `CC BY` 계열뿐이다. `CC BY-NC`, 모든 `-ND`, `Other`, custom, 누락, 만료된 COVID 임시 허용, author manuscript의 묵시적 상태는 제외한다. 논문별 `PMCID`, DOI, journal, copyright statement, 정확한 license URL, OA API의 `updated`, retraction 상태를 저장한다. 철회 논문, 사진·도표·캡션·supplement, 별도 credit가 있는 제3자 요소는 제외하고 JATS의 어문 본문만 사용한다.

한국어 query를 영문 evidence에 연결하면 cross-lingual retrieval replay가 된다. 생성 query는 원문에 없는 의학적 주장을 추가하면 안 되며, 개인 진단·처방 질문은 생성하지 않는다.

### 2.5 미국 연방정부 자료: public domain이라는 한 문장으로 승인하지 않는다

[17 U.S.C. §105](https://uscode.house.gov/view.xhtml?edition=2023&num=0&req=granuleid%3AUSC-2023-title17-section105)는 미국 정부 직원이 직무상 작성한 정부 저작물에 미국 저작권 보호가 적용되지 않는 원칙을 둔다. 그러나 계약자·수탁자·주정부 저작물, 정부가 이전받은 저작권, 문서 안의 제3자 요소, 미국 외 관할의 보호 가능성은 별개다. [GovInfo 정책](https://www.govinfo.gov/about/policies)도 정부 문서 안의 제3자 저작물은 자동 사용 허락이 아니라고 설명한다.

CDC는 [자료 이용 정책](https://www.cdc.gov/other/agencymaterials.html)에서 대부분의 정보가 public domain이라고 하면서도 출처표시, 비후원 고지, 실질 내용 불변, 무료 원문 위치 고지 및 제3자 예외를 요구한다. 수집은 공식 [CDC Content Services API](https://tools.cdc.gov/api/docs/info.aspx)의 다음 경로만 후보로 한다.

- 목록: `https://tools.cdc.gov/api/v2/resources/media?mediatypes=HTML&sourceacronym=CDC&max={N}&pagenum={PAGE}`
- 내용: `https://tools.cdc.gov/api/v2/resources/media/{MEDIA_ID}/content`

이 조건은 생성 query/evidence 재배포와의 관계를 별도 판단해야 하므로 `review_required`다. 문서가 CDC 직원 직접 작성임을 확인할 수 없거나 source/attribution/copyright 필드가 제3자를 가리키면 제외한다. 역외 배포와 “실질 내용 변경 금지” 검토가 끝나지 않으면 50K는 공공누리 30K + 법령 20K로 대체한다.

### 2.6 OpenDART: 사실 추출과 원문 재배포를 구분

[OpenDART 소개](https://opendart.fss.or.kr/intro/main.do)는 공시 원문 XML과 재무정보를 추출해 활용할 수 있다고 설명한다. 그러나 [이용약관 제16조](https://opendart.fss.or.kr/intro/terms.do)는 API·프로그램의 저작권을 금융감독원에 두고, 나머지는 저작권법과 공공데이터법에 따른다고 할 뿐 제출기업 원문의 공개 재배포·2차적 학습 데이터 배포를 일괄 허락하지 않는다.

- API 목록: <https://opendart.fss.or.kr/intro/infoApiList.do>
- 공시정보 가이드: <https://opendart.fss.or.kr/guide/main.do?apiGrpCd=DS001>
- 원문 API: `https://opendart.fss.or.kr/api/document.xml?crtfc_key={KEY}&rcept_no={RCEPT_NO}`

따라서 v1에서 공시 narrative, 표 전체, 주석, 감사보고서 문장을 재배포하지 않는다. 숫자·날짜·회사 코드 같은 비창작적 필드를 선별해 새 표현으로 만드는 방안도 DB 권리와 이용약관에 대한 서면 확인 전에는 `no_go`다. API key는 어떠한 산출물에도 넣지 않는다.

## 3. 제외 데이터와 benchmark 노출

### AIHub

[AIHub 이용정책](https://aihub.or.kr/intrcn/guid/usagepolicy.do?currMenu=151&topMenu=105)은 기본적으로 비상업적 연구·개발 활용을 설명하고, 데이터셋 판매 등 상업 이용은 수행기관과 별도 협의가 필요하다고 명시한다. 개별 데이터의 권리자와 조건도 다를 수 있다. 따라서 다운로드 가능하거나 모델 학습이 가능해도 **우리 공개 학습 데이터의 재배포 허락으로 보지 않는다**. 수행기관의 명시적 서면 허락과 downstream 재배포 조건을 확보한 특정 데이터만 향후 별도 shard로 재검토한다.

### KLUE, KorQuAD, MIRACL

- [KLUE 공식 저장소](https://github.com/KLUE-benchmark/KLUE)는 CC BY-SA 4.0이지만 공식 MTEB Korean의 KLUE-TC/STS 평가 원천이다.
- [KorQuAD 공식 사이트](https://korquad.github.io/)는 KorQuAD 2.x에 CC BY-ND 2.0 KR을 명시하며 SQuADKorV1 계열 평가와 직접 연결된다.
- [MIRACL 공식 저장소](https://github.com/project-miracl/miracl)는 annotations/repository를 Apache-2.0으로 표시하지만 corpus는 Wikipedia dump 파생이다. 한국어 query/qrel과 corpus 모두 공식 MTEB 및 Sionic 비교의 평가 원천이다.

권리 허용과 과학적 평가 적합성은 별개다. 이 세 데이터는 train/validation/negative mining/teacher prompt/example/few-shot에 넣지 않는다. 라이선스가 허용해도 benchmark 점수를 오염시키므로 `no_go_training`이다.

## 4. 100만 행 생성 파이프라인

### Stage 0 — 평가 봉인

이 단계는 source download, chunking, query 생성, teacher 호출보다 먼저 실행한다.

1. 공식 MTEB Korean v1 6개 과제와 Sionic-9의 모든 공개 split을 고정한다.
2. query, qrel, corpus, title, URL/article ID를 원본 revision과 함께 별도 read-only blocklist로 만든다.
3. normalized SHA-256, 5-gram MinHash, source-specific ID, benchmark query embedding fingerprint를 생성한다.
4. blocklist manifest의 hash를 모든 학습 행에 기록한다.
5. 새로운 사내 종합 평가셋도 처음 사용하기 전에 동일하게 봉인하고 이후 학습 시스템에서 읽기 전용으로 둔다.

평가 문서에서 query를 만든 다음 행만 삭제하는 것은 늦다. teacher가 평가 문서를 본 것 자체가 leakage이므로 모든 차단은 생성 전 corpus 단계에서 수행한다.

### Stage 1 — 권리와 revision 고정

source adapter는 본문보다 먼저 license evidence를 가져온다. 다음 중 하나라도 없으면 quarantine한다.

- 허용된 정확한 license ID와 근거 URL
- canonical source URL와 실제 content revision
- publisher/author 또는 공공기관 attribution
- raw content SHA-256와 수집시각
- third-party-rights, 개인정보, 철회·삭제 상태
- API 승인/이용목적과 호출 도구 revision

라이선스가 나중에 바뀌거나 source가 삭제될 수 있으므로 license page와 item metadata의 WARC/PDF/HTML 증거 hash를 내부 audit storage에 보존한다. 공개 데이터에는 민감한 원본 snapshot 대신 근거 URL과 hash만 넣는다.

### Stage 2 — 정제와 chunking

- HTML navigation, 광고, 댓글, reference list, boilerplate, OCR artifact를 분리한다.
- 개인정보·연락처·고유식별정보·사건 당사자 정보와 제3자 인용문을 제거한다.
- 128–512 token passage와 512–4,096 token long-context passage를 함께 만든다.
- 제목·절·조문·표제 관계를 보존하고, 숫자·단위·시점은 원문과 대조한다.
- exact hash → source ID → 5-gram MinHash 순으로 benchmark와 충돌하는 chunk를 제거한다.
- embedding near-duplicate는 자동 삭제가 아니라 높은 recall의 quarantine 후보로 쓰고 사람이 경계 사례를 확인한다.

### Stage 3 — grounded query/evidence 생성

한 evidence에서 query를 무작정 여러 개 뽑지 않는다. query 유형별 목표 분포를 둔다.

| 유형 | 비중 | 생성 원칙 |
|---|---:|---|
| 짧은 실제 검색형 | 30% | 핵심 entity/조건을 남기되 문장 복사를 피함 |
| 완전한 자연어 질문 | 20% | evidence만으로 답할 수 있어야 함 |
| 요건·예외·수치·시점 | 15% | 법령/공공 문서 구조와 정확히 연결 |
| long-document section retrieval | 15% | 문서 전체에서 특정 절을 찾게 함 |
| cross-lingual Korean→English | 10% | PMC/연방 문서 전용, 원문 근거 보존 |
| 구어체·오탈자·OCR·약어 | 5% | 의미를 바꾸지 않는 변형만 허용 |
| multi-hop | 5% | 서로 license-compatible한 두 evidence가 모두 필요 |

generator는 `query`, `answer`, `evidence_span`, `supporting_chunk_ids`, `claim_atoms`를 함께 출력한다. 로컬 또는 계약상 산출물 재배포가 가능한 teacher만 쓰며 model ID, exact revision, decoding, prompt hash를 기록한다. benchmark 예시를 teacher few-shot prompt로 사용하지 않는다.

### Stage 4 — 검증과 negative mining

모든 행은 generator와 다른 verifier가 다음을 검사한다.

- query의 모든 claim atom이 positive에 entail되는가
- positive 없이도 답이 노출되거나 query가 문장을 그대로 복사하지 않았는가
- 정답 passage가 하나 이상이며 multi-positive를 누락하지 않았는가
- 법률·의학 문장을 개인 자문처럼 바꾸지 않았는가
- query/answer가 원문 수치·시점·부정을 보존하는가

hard negative는 승인된 동일 shard의 문서에서 BM25와 현재 embedding model로 채굴한다. positive보다 너무 유사한 후보, verifier가 부분 정답으로 본 후보, 같은 문서의 중복 chunk는 negative로 쓰지 않고 secondary positive 또는 quarantine으로 보낸다. benchmark corpus는 candidate index 자체에 넣지 않는다.

검수 기준:

- 100%: 독립 teacher entailment + answerability + false-negative 검사
- 100%: hash/MinHash/ID benchmark 차단, license compatibility 검사
- 100%: 고위험 flag와 teacher 불일치 행 인간 판정
- 최소 10%: 법률·보건 행 층화 인간 표본
- 최소 2%: 나머지 source/query-type별 층화 인간 표본
- 100% 인간 검증: 5K release anchor set

표본 precision이 98% 미만이거나 source별 95% 미만이면 해당 batch 전체를 보류하고 생성 규칙을 수정한 뒤 다시 검사한다. 검수자는 정답성, 권리, 안전을 각각 구분해 기록한다.

## 5. 행·manifest 스키마

최소 행 스키마:

```json
{
  "row_id": "sha256:...",
  "query": "...",
  "positives": [{"chunk_id": "...", "text": "..."}],
  "negatives": [{"chunk_id": "...", "text": "...", "miner": "..."}],
  "source_id": "kogl_type1",
  "canonical_url": "https://...",
  "source_revision": "...",
  "content_sha256": "...",
  "license_id": "KOGL-1",
  "license_evidence_url": "https://...",
  "attribution_id": "attr:...",
  "transform_chain": ["html_to_text@...", "chunker@...", "querygen@..."],
  "generator_model_revision": "...",
  "verifier_model_revision": "...",
  "benchmark_blocklist_revision": "sha256:...",
  "release_status": "approved"
}
```

공개 데이터셋 루트에는 다음을 함께 배포한다.

- `SOURCES.jsonl`: 문서별 원문, publisher, revision, license evidence, content hash
- `ATTRIBUTION.md`와 기계판독 `ATTRIBUTION.jsonl`
- `TRANSFORMS.json`: parser/chunker/generator/verifier 정확한 revision과 prompt hash
- `BLOCKLIST_AUDIT.json`: benchmark 원문을 공개하지 않고 revision/hash/count와 충돌 제거 통계만 공개
- `REMOVALS.jsonl`: 삭제 요청을 반영할 tombstone ID와 이유, 원문은 미포함
- `DATASET_CARD.md`: shard별 라이선스, 알려진 한계, 개인정보·안전·오염 검사 결과

## 6. release gate

다음 조건을 모두 충족해야 공개한다.

- source adapter의 `status`가 `go` 또는 모든 `review_required`가 서면으로 해소된 `conditional_go`
- 문서별 license evidence와 revision이 100% 존재
- 허용 license 외 행 0건, source unknown 0건, secret/API key 0건
- benchmark exact/ID 충돌 0건, MinHash 충돌 0건, near-duplicate quarantine 미해결 0건
- 개인정보·제3자 권리·철회·삭제 flag 미해결 0건
- attribution manifest coverage 100%
- sample precision 기준 충족과 human audit 서명
- shard별 라이선스와 배포 패키지의 compatibility review 완료
- 공개 전 independent legal review와 model/dataset card의 비보증 문구 반영

조건을 충족하지 못한 batch를 모델 학습에 쓰는 것은 non-commercial 연구 트랙에서 별도로 결정할 수 있지만, 해당 모델과 adapter를 rights-safe/public-release 모델로 부르거나 공개 조직에 자동 업로드하면 안 된다.

## 7. 실제 시작 순서

1. `configs/data_sources_v1.json`을 validation schema로 삼아 source manifest validator를 먼저 구현한다.
2. 평가 blocklist를 고정하고 hash manifest를 서명한다.
3. 공공누리 0/1유형과 국가법령정보 adapter로 50K pilot을 만든다.
4. 5K 인간 검증 anchor와 20K blind dev를 먼저 만들고 학습 데이터와 조직·저장소를 분리한다.
5. Wikipedia share-alike shard와 PMC CC0/BY shard를 각각 추가한다.
6. CDC 법무 검토가 통과하지 않으면 fallback 50K로 즉시 대체한다.
7. 100K → 300K → 1M 순으로 quality/중복/도메인 분포를 점검하며 확장한다.

이 순서의 목적은 100만 행을 빨리 채우는 것이 아니라, 한 행을 역추적했을 때 원문·revision·권리 근거·생성 과정·검증·평가 차단 상태를 재현할 수 있게 하는 것이다.
