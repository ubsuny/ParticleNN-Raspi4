# ParticleNN-Raspi4

**See the Wiki for complete details as to what this project is!**

Instructions for Raspberry pi 4

To get this project running on a raspberry pi 4, you need to have Jupiter notebooks installed, python 3 (complete with Matplotlib and Numpy modules), TensorFlow, as well as a Raspbin OS. You will need a raspberry pi 4, its power supply cord, an ethernet cable and the necessary adaptors to fit the other end into your computer, a SD card and an SD card reader.

First things first, install this raspberry pi imager on your computer: https://www.raspberrypi.org/software/ This will allow you to download a OS (Operating system) onto your Raspberry pi 4. Once you have this installed, place the SD card in the SD card reader, plug it into your computer, and place the OS onto your Raspberry pi 4. Before disconnecting the SD card from your computer, place a blank file called ssh, with no file extension, onto the SD card. This is to enable ssh. Then, safely disconnect the SD card and place the SD card into your raspberry pi 4.

Next, wire up the raspberry pi 4. Plug the ethernet cable into the raspberry pi 4, and the other end into your computer. Wire up the raspberry pi 4 power supply into either your computer or an outlet, depending on which cable you have.

ssh into your raspberry pi 4. Open a terminal, and type ssh pi@raspberrypi.local The password is raspberry, and congratulations, you're in!

To make things easier, using VNC viewer enables you to view the desktop for your raspberry pi 4, below are instructions for enabling it.

Enable VNC viewer on your raspberry pi 4. Once your ssh'ed in, run these lines of code:
sudo apt update

sudo apt install realvnc-vnc-server realvnc-vnc-viewer

sudo raspi-config

Then navigate to Interfacing Options > VNC and select Yes, hit enter and finish. To avoid a common problem with a blank desktop, also change the resolution to max using sudo raspi-config

Be sure to reboot using: sudo reboot

Install VNC viewer from: https://www.realvnc.com/en/connect/download/viewer/

Using VNC viewer, connect to 'pi@raspberrypi.local' using password raspberry. Congratulations, you're in again! You can also connect to wifi easily using VNC viewer, just click the wifi button on the desktop bar.

For the purposes of this code, you can use either an ssh terminal or a VNC viewer terminal.

Install python 3 using these lines of code:
sudo apt update

sudo apt install python3 idle3

Install Jupiter notebook using this tutorial: https://www.instructables.com/Jupyter-Notebook-on-Raspberry-Pi/
Reboot using: sudo reboot

Install TensorFlow, the shortcut is fine, from this tutorial: https://qengineering.eu/install-tensorflow-2.1.0-on-raspberry-pi-4.html

Now your all set to run this code on your raspberry pi 4! Simply run these lines of code:

git clone https://github.com/ubsuny/ParticleNN-Raspi4.git

jupyter-notebook

Navigate to the .ipynb file, and click run all!
