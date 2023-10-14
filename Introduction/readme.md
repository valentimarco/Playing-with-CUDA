## Install CUDA
To install cuda follow the official docs or from AUR if you are in ArchLinux.

## Install libs
You need to install GLUT and Glu. (In ArchLinux you can install it from AUR)


## List of Fixes

1. fix symlink after installation (this is for WSL)

    There is another solution works for me :
    
    - Open cmd as Administrator and cd into C:\Windows\System32\lxss\lib
    - Delete libcuda.so and libcuda.so.1 (You can also do this in Windows Explorer as well)
    - Run wsl -e /bin/bash in cmd and you should already in /mnt/c/Windows/System32/lxss/lib, now you have permission to create symlink:
     ``
        ln -s libcuda.so.1.1 libcuda.so.1
        ln -s libcuda.so.1.1 libcuda.so
      ``



