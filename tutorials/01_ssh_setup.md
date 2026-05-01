# Jetson Orin Nano — SSH 연결 세팅 메뉴얼

---

## 작성자
**Author**: Hyun-seo In  
**Affiliation**: Hanyang Univ. Dept. of BME (major), AE (minor)  
**Contact**: inhsroy@hanyang.ac.kr  
**GitHub**: https://github.com/InHyunseo  
**Date**: 2026.05

---

## 목차
1. [사전 준비](#1-사전-준비)
2. [Jetson 측 확인](#2-jetson-측-확인)
3. [노트북 측 설정 (Windows)](#3-노트북-측-설정-windows)
4. [SSH 연결](#4-ssh-연결)
5. [Wi-Fi SSH 연결](#5-wi-fi-ssh-연결)
6. [VS Code Remote SSH 연동](#6-vs-code-remote-ssh-연동)
7. [트러블슈팅](#7-트러블슈팅)

---

## 1. 사전 준비

- Jetson Orin Nano에 JetPack 설치 완료 상태
- **Jetson 전원 어댑터 연결 필수** — USB-C는 데이터 전용, 전원 공급 불가
- 데이터 통신 지원 USB-C 케이블

연결 순서:
1. Jetson 전원 어댑터 연결 → 부팅 완료 대기
2. USB-C 케이블로 Jetson ↔ 노트북 연결

---

## 2. Jetson 측 확인

### USB device mode 확인

JetPack 설치 시 `nv-l4t-usb-device-mode` 관련 스크립트가 자동으로 포함됨.
USB-C 케이블 연결 시 `l4tbr0` 브리지 인터페이스에 자동으로 `192.168.55.1` 할당됨.

```bash
# USB-C 연결 후 IP 확인
ip addr show l4tbr0
# inet 192.168.55.1/24 출력 확인
```

> ⚠️ USB-C 케이블 연결 전에는 l4tbr0 상태가 DOWN으로 표시됨
> 케이블 연결 후 UP 되면서 inet 주소가 뜨는 게 정상
> `g_ether` 모듈이나 별도 서비스 실행 불필요 — JetPack이 자동 처리

### SSH 서버 확인

```bash
sudo systemctl status ssh
# active (running) 확인
```

---

## 3. 노트북 측 설정 (Windows)

### 1. 장치 인식 확인

장치 관리자 → 네트워크 어댑터 → **Remote NDIS Compatible Device** 확인

> ⚠️ 이 항목이 없으면 USB-C 케이블이 데이터 미지원이거나 연결 문제

### 2. IPv4 수동 설정

**Windows 키 + R** → `ncpa.cpl` → 엔터

네트워크 어댑터 목록에서 `Remote NDIS Compatible Device` 우클릭 → **속성**
→ **인터넷 프로토콜 버전 4 (TCP/IPv4)** 더블클릭

```
● 다음 IP 주소 사용 선택
IP 주소:      192.168.55.100
서브넷 마스크:  255.255.255.0
기본 게이트웨이: 192.168.55.1
```

확인 클릭

### 3. 연결 테스트

cmd에서:
```
ping 192.168.55.1
```
응답 오면 성공

---

## 4. SSH 연결

### 처음 연결 시

```
ssh <username>@192.168.55.1
```

`Are you sure you want to continue connecting (yes/no)?` → `yes` 입력
비밀번호 입력 후 접속 완료

### known_hosts 오류 발생 시

```
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED!
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
```

이전 접속 기록 충돌 → 아래 명령으로 기록 삭제 후 재접속:

```
ssh-keygen -R 192.168.55.1
ssh <username>@192.168.55.1
```

---

## 5. Wi-Fi SSH 연결

Jetson에서 Wi-Fi 연결 (USB-C SSH 접속한 상태에서):
```bash
# 주변 Wi-Fi 스캔
nmcli device wifi list

# Wi-Fi 연결
sudo nmcli device wifi connect "SSID명" password "비밀번호"

# 연결된 IP 확인
ifconfig
# wlP~ 항목 맨 하단의 inet 주소 확인
```

노트북에서 접속:
```
ssh <username>@<jetson_wifi_ip>
```

> ⚠️ Wi-Fi IP는 DHCP라 재부팅 시 바뀔 수 있음
> 공유기 설정에서 MAC 주소 기반 IP 고정 권장

---

## 6. VS Code Remote SSH 연동

### 1. 확장 프로그램 설치

VS Code → Extensions (`Ctrl + Shift + X`) → `Remote - SSH` 검색 → **Install**

> Microsoft 공식 확장 프로그램 사용

### 2. SSH Host 추가

`Ctrl + Shift + P` → `Remote-SSH: Add New SSH Host` 선택

#### USB-C 연결용

입력란에 다음 입력 후 엔터:
```
ssh <username>@192.168.55.1
```

config 파일 선택 창이 뜨면 → `C:\Users\<username>\.ssh\config` 선택

#### Wi-Fi 연결용

같은 방식으로 추가:
```
ssh <username>@<jetson_wifi_ip>
```

### 3. 연결

1. `Ctrl + Shift + P`
2. `Remote-SSH: Connect to Host` 선택
3. 추가한 호스트 (예: `192.168.55.1` 또는 `<jetson_wifi_ip>`) 선택
4. 처음 연결 시 platform 선택 → **Linux** 선택
5. 비밀번호 입력 후 연결 완료

### 4. config 파일 직접 편집 (선택)

호스트 이름을 알아보기 쉽게 바꾸고 싶으면 `C:\Users\<username>\.ssh\config` 직접 편집:

```
Host jetson-usb
    HostName 192.168.55.1
    User <username>
    Port 22

Host jetson-wifi
    HostName <jetson_wifi_ip>
    User <username>
    Port 22
```

이러면 호스트 목록에 `jetson-usb`, `jetson-wifi`로 깔끔하게 표시됨.
Wi-Fi IP가 바뀌면 `HostName` 한 줄만 수정하면 됨.

---

## 7. 트러블슈팅

| 증상 | 원인 | 해결 |
|------|------|------|
| `Remote NDIS Compatible Device` 없음 | USB-C 미인식 | 데이터 지원 케이블 확인, 재연결 |
| `ping` 불통 | IPv4 설정 오류 | ncpa.cpl에서 IP 재설정 확인 |
| `l4tbr0`에 inet 없음 | USB-C 미연결 | 케이블 연결 후 재확인 |
| `WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED` | known_hosts 충돌 | `ssh-keygen -R <ip>` 실행 후 재접속 |
| `Bad owner or permissions on ...\.ssh\config` | config 파일 권한 문제 | 관리자 권한 cmd에서 아래 명령 실행 |
| SSH 연결 거부 | SSH 서버 미실행 | `sudo systemctl start ssh` |
| VS Code "Could not establish connection" | VS Code Server 설치 꼬임 | Jetson에서 `rm -rf ~/.vscode-server` 후 재연결 |
| Wi-Fi IP 바뀜 | DHCP | 공유기에서 MAC 기반 IP 고정 |

### config 파일 권한 오류 해결

`Bad owner or permissions` 오류 발생 시, 관리자 권한 cmd에서:

```cmd
icacls C:\Users\<username>\.ssh\config /inheritance:r
icacls C:\Users\<username>\.ssh\config /grant:r "%username%:F"
```

---
> 실제 환경: Jetson Orin Nano dev kit / JetPack 6.x / Windows 11 (WSL2)