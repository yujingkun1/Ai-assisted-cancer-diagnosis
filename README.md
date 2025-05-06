# 癌影云析-基于多模态数据融合的癌症辅助诊断系统

## Windows部署

用户需要下载Python，Mysql，Anocanda等依赖软件，在完成相关依赖的配置后，最后通过浏览器访问到系统界面。 安装完成后的软件预计会占用5000端口，请提前预留端口以避免冲突。

具体步骤如下：

### 部署Anaconda

-   访问 Anaconda 官方网站（<https://www.anaconda.com/>）。
-   根据你的操作系统（Windows）选择合适的安装程序。Anaconda 提供了包含众多数据科学和机器学习相关库的 Python 发行版。
-   运行安装程序，按照安装向导的提示完成安装。在安装过程中，你可以选择将conda Ana 添加到系统环境变量（推荐勾选），这样可以在命令提示符（CMD）等工具中方便地使用 conda 命令。

### 创建虚拟环境

-   在 Windows 操作系统中，可以通过在 “开始” 菜单中搜索 “命令提示符” 或 “Anaconda Prompt” 来打开它。如果安装过程中添加了环境变量，“Anaconda Prompt” 会自动配置好 conda 的环境，方便使用。
-   输入`conda create -n myenv python=2.9.0`。
-   在执行这个命令后，conda 会从其仓库下载所需的 Python 版本以及相关的依赖包，并将它们安装到新创建的虚拟环境中。

### 激活虚拟环境

-   在命令提示符或 Anaconda Prompt 中，使用`conda activate myenv`命令来激活虚拟环境。激活成功后，你会在命令提示符或 Anaconda Prompt 的提示符前面看到环境名称`(myenv)`，这表示你已经进入该虚拟环境。

### 在虚拟环境中安装包

-   输入`pip install -r requirements.txt`安装相关依赖

### 启动程序

-   打开终端，切换到项目文件夹

-   输入`python app.py`运行程序

-   成功运行后打开Flask 会启动一个开发服务器，监听本地的 `5000` 端口。可以在浏览器中访问 `http://127.0.0.1:5000` 来查看运行结果。

## Linux部署

在 Linux 系统下可以使用命令行来部署

### 部署Anaconda

- 在终端中使用`wget`命令下载 Anaconda 安装脚本，输入

  ```         
  wget https://repo.anaconda.com/archive/Anaconda3-2024.06-1-Linux-x86_64.sh
  ```

- 对下载的脚本文件赋予执行权限，然后运行安装脚本：

  ```         
  chmod +x Anaconda3-2024.06-1-Linux-x86_64.sh ./Anaconda3-2024.06-1-Linux-x86_64.sh
  ```

#### **配置环境变量**

安装完成后，运行以下命令使 Anaconda 的环境变量生效：

```         
source ~/.bashrc
```

### 创建虚拟环境

- 在Linux系统下，按住CTRL+ALT+R打开终端，

- 使用以下命令创建一个指定 Python 版本的虚拟环境：

  ```         
  conda create --name myenv python=2.9.0
  ```

### **激活虚拟环境**

-   使用`conda activate myenv`命令来激活虚拟环境。激活成功后，你会在终端用户名前面看到环境名称`(myenv)`，这表示你已经进入该虚拟环境。

### 在虚拟环境中安装包

-   输入`pip install -r requirements.txt`安装相关依赖

### 启动程序

-   打开终端，切换到项目文件夹

-   输入`python app.py`运行程序

-   成功运行后打开Flask 会启动一个开发服务器，监听本地的 `5000` 端口。可以在浏览器中访问 [`http://127.0.0.1:5000`](http://127.0.0.1:5000) 来查看运行结果。