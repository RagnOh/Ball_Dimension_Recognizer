#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
	VideoCapture cap(0); //inizzializza connessione alla video camera

	if (!cap.isOpened())  // se non riesce ad aprire la camera stampa il messaggio
	{
		cout << "Errore apertura camera!" << endl;
		return -1;
	}

	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;

	int c;

	while (true)
	{
		Mat src;
		Mat grad;
		Mat src_gray;

		bool bSuccess = cap.read(src); // legge un nuovo frame dalla telecamera

		if (!bSuccess) //se non ci riesce stampa il messaggio di errore ed interrompe il loop
		{
			cout << "Errore lettura frame video" << endl;
			break;
		}
		GaussianBlur(src, src, Size(3, 3), 0, 0, BORDER_DEFAULT);

		/// Converto i colori della camera in una scala di grigi in modo da semplificare il lavoro
		cvtColor(src, src_gray, cv::COLOR_BGR2GRAY);



		/// Genero grad_x e grad_y
		Mat grad_x, grad_y;
		Mat abs_grad_x, abs_grad_y;

		/// Gradiente X
		Sobel(src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
		convertScaleAbs(grad_x, abs_grad_x);

		/// Gradiente Y
		Sobel(src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
		convertScaleAbs(grad_y, abs_grad_y);

		/// Somma approssimata dei gradienti
		addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

		vector<Vec3f> circles;
		HoughCircles(grad, circles, HOUGH_GRADIENT, 1,
			src_gray.rows / 16,  // Cambiando questo valore si cambia il range di distanza voluto tra i cerchi
			100, 30, 1, 30 // cambiando gli ultimi 2 parametri
	   // (min_radius & max_radius) allargo o restringo il diametro di ricerca dei cerchi
		);
		for (size_t i = 0; i < circles.size(); i++)
		{
			Vec3i c = circles[i];
			Point center = Point(c[0], c[1]);
			// Centro del cerchio
			circle(src, center, 1, Scalar(0, 100, 100), 3, LINE_AA);
			// Perimetro cerchio
			int radius = c[2];
			circle(src, center, radius, Scalar(255, 0, 255), 3, LINE_AA);
		}
		imshow("Cerchi identificati", src);



		if (waitKey(5) == 27) //attendi che l' utente prema 'ESC' per 27ms ed esci
		{
			cout << "esc key premuto dall' utente" << endl;
			break;
		}
	}

	return 0;

}