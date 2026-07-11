# 110 — Sionic AutoRAG domain adaptation

F2 finance/banking/commerce/legal 100K를 current-student hard negative로 재채굴하고
general 1M과 50:50 replay한다.

- AutoRAG query/corpus exact overlap 0
- 전체 15-task query/evaluation-text critical overlap 0
- 영어·중국어 domain signal의 한국어 cross-lingual transfer 검증
- AutoRAG 단일 점수가 아니라 Sionic 9 macro/공식 Korean/clean 회귀로 승격
- 파생 dataset과 merged model을 `target-adapted-autorag-domain`으로 공개

실행: [`scripts/run_sionic_autorag_adaptation_queue.sh`](../../scripts/run_sionic_autorag_adaptation_queue.sh)

설계·수치: [`docs/27_SIONIC_AUTORAG_DOMAIN_ADAPTATION.md`](../../docs/27_SIONIC_AUTORAG_DOMAIN_ADAPTATION.md)
