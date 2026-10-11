# 호스팅 자동 검증 상태

[English](ci-status.md) | [한국어](ci-status.ko.md)

저장소에는 읽기 권한만 사용하는 Windows와 Linux CPU 검증 워크플로를 구성했습니다. 기존 간이 워크플로는 현재 검사에 필요한 의존성을 설치하지 않아 삭제했습니다.

2026년 10월 11일 공개한 main 브랜치에 `Verify portable CPU code` 실행을 요청했으나 GitHub가 HTTP 422와 `Actions has been disabled for this user.`라는 사유로 거절했습니다. 실행은 생성되지 않았으며 GitHub Actions를 통과했다고 주장하지 않습니다. 로컬 검사와 분리된 환경 검증은 [결과](results.ko.md)와 [환경 검증](environment-verification.ko.md)에 기록했습니다.

인증한 기존 저장소에는 게시가 완료됐습니다. 비로그인 HTTP 확인은 404를 반환했으며 원인을 추측하지 않습니다. GitHub 접근 없이 코드와 문서를 확인할 수 있도록 공개 사이트에서 [소스 ZIP](https://park-dongmyung.com/downloads/korean-nlp-rag-source.zip)을 제공합니다.
