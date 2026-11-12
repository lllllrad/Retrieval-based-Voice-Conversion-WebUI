# RVC 서버 관리. 서비스: webui (7865), live (7867). 인자를 생략하면 전체.
# PID: ~/.local/run/rvc/<서비스>.pid   로그: ~/.local/state/rvc/<서비스>.log

svc := justfile_directory() / "scripts/rvc-svc"

# 상태 보기
default: status

# 서버 시작        예) just up / just up live
up *services:
    @{{svc}} up {{services}}

# 서버 중지        예) just down / just down webui
down *services:
    @{{svc}} down {{services}}

# 서버 재시작      예) just restart live
restart *services:
    @{{svc}} restart {{services}}

# 서버 상태
status:
    @{{svc}} status

# 로그 보기        예) just logs live / just logs live -f
logs service *flags:
    @{{svc}} logs {{service}} {{flags}}
