# ROS 2 Humble + Ignition Fortress 설치 메뉴얼

**Author**: Hyun-seo In (Hanyang Univ. BME/AE) — <inhsroy@hanyang.ac.kr>

ROS 2 워크스페이스를 처음 설치하고 빌드하는 절차.

기준 경로:
- 저장소 루트: `~/AUE4040`
- ROS 2 워크스페이스: `~/AUE4040/ros2_ws`

## 0. (Windows 사용자) WSL2 + Ubuntu 22.04 설치

Windows에서 작업하는 경우 먼저 WSL2에 Ubuntu 22.04를 설치한다.

**방법 1 — Microsoft Store**
1. Microsoft Store 실행
2. "Ubuntu 22.04 LTS" 검색 후 설치
3. 실행 → 사용자명/비밀번호 설정

**방법 2 — PowerShell (관리자)**

```powershell
wsl --install -d Ubuntu-22.04
```

설치 후 확인:

```powershell
wsl -l -v          # 설치된 배포판 / WSL 버전 확인
wsl --set-default-version 2
```

이후 모든 명령은 WSL Ubuntu 터미널 안에서 실행한다.

## 1. 대상 환경

- Ubuntu 22.04 (네이티브 또는 WSL2)
- ROS 2 Humble
- Ignition Fortress (`ign gazebo`)

## 2. ROS 2 Humble 설치

```bash
sudo apt update
sudo apt install software-properties-common curl -y
sudo add-apt-repository universe

export ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F\" '{print $4}')
curl -L -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo ${UBUNTU_CODENAME:-${VERSION_CODENAME}})_all.deb"
sudo dpkg -i /tmp/ros2-apt-source.deb

sudo apt update && sudo apt upgrade -y
sudo apt install ros-humble-desktop ros-dev-tools -y
```

셸 환경 + rosdep:

```bash
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source /opt/ros/humble/setup.bash
sudo rosdep init
rosdep update
```

## 3. Ignition Fortress 설치

```bash
sudo apt-get install lsb-release gnupg curl -y

sudo curl https://packages.osrfoundation.org/gazebo.gpg \
  --output /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] https://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" \
  | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null

sudo apt-get update
sudo apt-get install ignition-fortress -y
ign gazebo --versions   # 확인
```

## 4. 워크스페이스 준비

```bash
mkdir -p ~/AUE4040/ros2_ws/src
cd ~/AUE4040/ros2_ws/src
# 필요한 패키지 저장소를 여기에 clone
# git clone <repo-url>
```

서브모듈이 있다면:

```bash
cd <패키지 경로>
git submodule sync --recursive
git submodule update --init --recursive
```

## 5. 의존성 설치

```bash
cd ~/AUE4040/ros2_ws
source /opt/ros/humble/setup.bash
rosdep install --from-paths src --ignore-src -r -y
```

## 6. 빌드

```bash
cd ~/AUE4040/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

특정 패키지만 빌드:

```bash
colcon build --packages-select <패키지명>
```

## 7. 빌드 후

```bash
source ~/AUE4040/ros2_ws/install/setup.bash
ros2 pkg list        # 패키지 확인
ros2 node list       # 실행 중 노드 확인
```
