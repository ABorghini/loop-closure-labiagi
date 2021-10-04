# Loop closure - LABIAGI

Per eseguire qualsiasi file è necessario aprire il terminale dalla cartella catwins_ws e lanciare i seguenti comandi:
source devel/setup.bash
catkin_make
roscore

Aprire poi un altro terminale per eseguire il nodo ros.
Per raccogliere le immagini:
rosrun video_node images.py

Per generare la bag of words, e altri file:
rosrun video_node bovw_generator.py

Per calcolare le performance:
rosrun video_node simili.py

###Autore: Alessia Borghini