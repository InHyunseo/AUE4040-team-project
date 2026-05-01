# Linux 기초 명령어 메뉴얼

**Author**: Hyun-seo In (Hanyang Univ. BME/AE) — <inhsroy@hanyang.ac.kr>

Jetson Orin Nano (Ubuntu) 환경에서 자주 쓰는 기본 명령어 정리.

---

## 1. 파일 / 디렉토리 탐색

| 명령어 | 설명 |
|---|---|
| `pwd` | 현재 작업 디렉토리 출력 |
| `ls` | 현재 디렉토리 파일 목록 |
| `ls -al` | 숨김 파일 포함, 상세 정보 표시 |
| `cd <경로>` | 디렉토리 이동 |
| `cd ..` | 상위 디렉토리로 이동 |
| `cd ~` | 홈 디렉토리로 이동 |
| `tree` | 디렉토리 구조를 트리 형태로 출력 |

## 2. 파일 / 디렉토리 조작

| 명령어 | 설명 |
|---|---|
| `mkdir <폴더명>` | 새 디렉토리 생성 |
| `mkdir -p a/b/c` | 중간 경로 포함 디렉토리 생성 |
| `touch <파일명>` | 빈 파일 생성 |
| `cp <원본> <대상>` | 파일 복사 |
| `cp -r <폴더> <대상>` | 디렉토리 재귀 복사 |
| `mv <원본> <대상>` | 파일/폴더 이동 또는 이름 변경 |
| `rm <파일>` | 파일 삭제 |
| `rm -r <폴더>` | 디렉토리 삭제 (재귀) |
| `rm -rf <폴더>` | 강제 삭제 (주의!) |

## 3. 파일 내용 보기 / 편집

| 명령어 | 설명 |
|---|---|
| `cat <파일>` | 파일 전체 출력 |
| `head -n 10 <파일>` | 앞부분 10줄 출력 |
| `tail -n 10 <파일>` | 끝부분 10줄 출력 |
| `tail -f <파일>` | 실시간 로그 모니터링 |
| `less <파일>` | 페이지 단위로 보기 (`q`로 종료) |
| `nano <파일>` | nano 에디터로 편집 |
| `vim <파일>` | vim 에디터로 편집 |

## 4. 권한 / 소유권

| 명령어 | 설명 |
|---|---|
| `chmod +x <파일>` | 실행 권한 부여 |
| `chmod 755 <파일>` | 권한을 숫자로 지정 |
| `chown user:group <파일>` | 소유자 변경 |
| `sudo <명령>` | 관리자 권한으로 실행 |

## 5. 검색

| 명령어 | 설명 |
|---|---|
| `find . -name "*.py"` | 현재 폴더 이하에서 파일명 검색 |
| `grep "문자열" <파일>` | 파일 내 문자열 검색 |
| `grep -r "문자열" .` | 디렉토리 재귀 검색 |
| `which <명령어>` | 실행 파일 경로 확인 |

## 6. 프로세스 / 시스템

| 명령어 | 설명 |
|---|---|
| `ps aux` | 실행 중인 프로세스 목록 |
| `top` / `htop` | 실시간 시스템 모니터링 |
| `kill <PID>` | 프로세스 종료 |
| `kill -9 <PID>` | 강제 종료 |
| `df -h` | 디스크 용량 확인 |
| `du -sh <폴더>` | 폴더 용량 확인 |
| `free -h` | 메모리 사용량 확인 |
| `uname -a` | 커널/시스템 정보 |

## 7. 네트워크

| 명령어 | 설명 |
|---|---|
| `ifconfig` / `ip addr` | 네트워크 인터페이스 확인 |
| `ping <주소>` | 네트워크 연결 테스트 |
| `ssh user@host` | SSH 원격 접속 |
| `scp <파일> user@host:<경로>` | 파일 원격 복사 |
| `wget <URL>` | 파일 다운로드 |
| `curl <URL>` | URL 요청 |

## 8. 패키지 관리 (Ubuntu)

| 명령어 | 설명 |
|---|---|
| `sudo apt update` | 패키지 목록 갱신 |
| `sudo apt upgrade` | 설치된 패키지 업그레이드 |
| `sudo apt install <패키지>` | 패키지 설치 |
| `sudo apt remove <패키지>` | 패키지 제거 |

## 9. Git 명령어

### 저장소 초기화 / 클론
| 명령어 | 설명 |
|---|---|
| `git init` | 현재 폴더를 git 저장소로 초기화 |
| `git clone <URL>` | 원격 저장소 복제 |
| `git remote -v` | 원격 저장소 목록 확인 |
| `git remote add origin <URL>` | 원격 저장소 등록 |

### 상태 / 변경사항 확인
| 명령어 | 설명 |
|---|---|
| `git status` | 현재 변경사항 상태 |
| `git diff` | 수정된 내용 비교 |
| `git log` | 커밋 히스토리 |
| `git log --oneline` | 한 줄로 간략히 |

### 커밋 / 푸시
| 명령어 | 설명 |
|---|---|
| `git add <파일>` | 변경사항 스테이징 |
| `git add .` | 전체 변경사항 스테이징 |
| `git commit -m "메시지"` | 커밋 생성 |
| `git push origin <브랜치>` | 원격 저장소에 푸시 |
| `git pull` | 원격 변경사항 가져오기 |
| `git fetch` | 원격 정보만 갱신 (병합 X) |

### 브랜치
| 명령어 | 설명 |
|---|---|
| `git branch` | 브랜치 목록 |
| `git branch <이름>` | 새 브랜치 생성 |
| `git checkout <브랜치>` | 브랜치 전환 |
| `git checkout -b <이름>` | 브랜치 생성 후 전환 |
| `git merge <브랜치>` | 브랜치 병합 |
| `git branch -d <이름>` | 브랜치 삭제 |

### 되돌리기
| 명령어 | 설명 |
|---|---|
| `git restore <파일>` | 변경사항 되돌리기 |
| `git reset HEAD <파일>` | 스테이징 취소 |
| `git reset --hard HEAD` | 모든 변경사항 폐기 (주의!) |
| `git revert <커밋해시>` | 특정 커밋을 취소하는 새 커밋 생성 |

### 사용자 설정
| 명령어 | 설명 |
|---|---|
| `git config --global user.name "이름"` | 사용자 이름 설정 |
| `git config --global user.email "메일"` | 이메일 설정 |

---

## 10. Jetson 보드 관련 명령어

### 시스템 정보 / 모니터링
| 명령어 | 설명 |
|---|---|
| `jetson_release` | Jetson 보드/JetPack 버전 정보 |
| `jtop` | Jetson 전용 시스템 모니터 (jetson-stats 설치 필요) |
| `tegrastats` | CPU/GPU/메모리/전력 실시간 모니터링 |
| `nvidia-smi` | GPU 상태 확인 (일부 Jetson 미지원) |
| `cat /etc/nv_tegra_release` | L4T(BSP) 버전 확인 |

### 전원 / 성능 모드
| 명령어 | 설명 |
|---|---|
| `sudo nvpmodel -q` | 현재 전원 모드 확인 |
| `sudo nvpmodel -m <번호>` | 전원 모드 변경 (0 = MAXN) |
| `sudo jetson_clocks` | 클럭을 최대치로 고정 |
| `sudo jetson_clocks --show` | 현재 클럭 상태 출력 |

### CUDA / TensorRT
| 명령어 | 설명 |
|---|---|
| `nvcc --version` | CUDA 컴파일러 버전 확인 |
| `dpkg -l | grep TensorRT` | TensorRT 패키지 확인 |
| `trtexec --onnx=model.onnx --saveEngine=model.engine` | ONNX → TensorRT 엔진 변환 |
| `trtexec --loadEngine=model.engine` | 엔진 추론 벤치마크 |

### 카메라 / 디바이스
| 명령어 | 설명 |
|---|---|
| `ls /dev/video*` | 연결된 카메라 디바이스 확인 |
| `v4l2-ctl --list-devices` | V4L2 카메라 목록 |
| `lsusb` | USB 디바이스 목록 |
| `i2cdetect -y -r 1` | I2C 디바이스 스캔 |

### 설치 / 업데이트
| 명령어 | 설명 |
|---|---|
| `sudo apt install nvidia-jetpack` | JetPack 메타패키지 설치 |
| `sudo pip3 install -U jetson-stats` | jtop 설치 |

---

## 11. 기타 유용한 명령어

| 명령어 | 설명 |
|---|---|
| `clear` | 터미널 화면 지우기 |
| `history` | 명령어 히스토리 |
| `echo "문자열"` | 문자열 출력 |
| `man <명령어>` | 명령어 메뉴얼 보기 |
| `<명령어> --help` | 간단한 도움말 |
| `Ctrl + C` | 실행 중인 명령 중단 |
| `Ctrl + D` | 터미널 종료 / EOF |
| `Tab` | 자동완성 |
