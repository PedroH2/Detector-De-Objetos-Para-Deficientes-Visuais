/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.detection;

import android.content.SharedPreferences;
import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Paint.Style;
import android.graphics.Picture;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.graphics.drawable.PictureDrawable;
import android.media.ImageReader.OnImageAvailableListener;
import android.media.MediaPlayer;
import android.os.Bundle;
import android.os.SystemClock;
import android.os.Vibrator;
import android.util.Size;
import android.util.TypedValue;
import android.view.View;
import android.webkit.WebSettings;
import android.widget.Button;
import android.widget.CompoundButton;
import android.widget.SeekBar;
import android.widget.Switch;
import android.widget.TextView;
import android.widget.Toast;
import android.widget.ToggleButton;

import java.io.IOException;
import java.util.LinkedList;
import java.util.List;
import org.tensorflow.lite.examples.detection.customview.OverlayView;
import org.tensorflow.lite.examples.detection.customview.OverlayView.DrawCallback;
import org.tensorflow.lite.examples.detection.env.BorderedText;
import org.tensorflow.lite.examples.detection.env.ImageUtils;
import org.tensorflow.lite.examples.detection.env.Logger;
import org.tensorflow.lite.examples.detection.tflite.Classifier;
import org.tensorflow.lite.examples.detection.tflite.TFLiteObjectDetectionAPIModel;
import org.tensorflow.lite.examples.detection.tracking.MultiBoxTracker;
import org.w3c.dom.Text;

import static org.checkerframework.checker.units.UnitsTools.A;

/**
 * An activity that uses a TensorFlowMultiBoxDetector and ObjectTracker to detect and then track
 * objects.
 */
public class DetectorActivity extends CameraActivity implements OnImageAvailableListener {
  private static final Logger LOGGER = new Logger();

  // Configuration values for the prepackaged SSD model.

  private static final int TF_OD_API_INPUT_SIZE = 300;
  private static final boolean TF_OD_API_IS_QUANTIZED = true;
  private static final String TF_OD_API_MODEL_FILE = "detect.tflite";
  private static final String TF_OD_API_LABELS_FILE = "labelmap.txt";
  private static final DetectorMode MODE = DetectorMode.TF_OD_API;

  // Minimum detection confidence to track a detection.

  private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.5f;
  private static final boolean MAINTAIN_ASPECT = false;
  private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);
  private static final boolean SAVE_PREVIEW_BITMAP = false;
  private static final float TEXT_SIZE_DIP = 10;
  OverlayView trackingOverlay;
  private Integer sensorOrientation;

  private Classifier detector;

  private long lastProcessingTimeMs;
  private Bitmap rgbFrameBitmap = null;
  private Bitmap croppedBitmap = null;
  private Bitmap cropCopyBitmap = null;

  private boolean computingDetection = false;

  private long timestamp = 0;

  private Matrix frameToCropTransform;
  private Matrix cropToFrameTransform;

  private MultiBoxTracker tracker;

  private BorderedText borderedText;

  private SeekBar seekBar;
  private MediaPlayer mediaPlayer;
  private Switch switch1;
  private TextView textStatus;
  private TextView textMetros;
  private static final String METROS_PREFERENCIA = "metrosPreferencia";
  private boolean b;

  @Override
  public void onPreviewSizeChosen(final Size size, final int rotation) {
    final float textSizePx =
        TypedValue.applyDimension(
            TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
    borderedText = new BorderedText(textSizePx);
    borderedText.setTypeface(Typeface.MONOSPACE);

    tracker = new MultiBoxTracker(this);

    int cropSize = TF_OD_API_INPUT_SIZE;

    try {
      detector =
          TFLiteObjectDetectionAPIModel.create(
              getAssets(),
              TF_OD_API_MODEL_FILE,
              TF_OD_API_LABELS_FILE,
              TF_OD_API_INPUT_SIZE,
              TF_OD_API_IS_QUANTIZED);
      cropSize = TF_OD_API_INPUT_SIZE;
    } catch (final IOException e) {
      e.printStackTrace();
      LOGGER.e(e, "Exception initializing classifier!");
      Toast toast =
          Toast.makeText(
              getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
      toast.show();
      finish();
    }

    previewWidth = size.getWidth();
    previewHeight = size.getHeight();

    sensorOrientation = rotation - getScreenOrientation();
    LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

    LOGGER.i("Initializing at size %dx%dc", previewWidth, previewHeight);
    rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
    croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Config.ARGB_8888);

    frameToCropTransform =
        ImageUtils.getTransformationMatrix(
            previewWidth, previewHeight,
            cropSize, cropSize,
            sensorOrientation, MAINTAIN_ASPECT);
    cropToFrameTransform = new Matrix();
    frameToCropTransform.invert(cropToFrameTransform);

    switch1 = findViewById(R.id.switch1);
    seekBar = findViewById(R.id.seekBarScala);
    textStatus = findViewById(R.id.textStatus);
    textMetros = findViewById(R.id.textMetros);;

    recuperarPreferencia();
    seekBar.setEnabled(false);

    seekBar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
      //Chamo a classe setOnSeekBarChangeListener que retorna alguns metodos para podermos utilizar
      @Override
      public void onProgressChanged(SeekBar seekBar, int i, boolean b) {

      }

      @Override
      public void onStartTrackingTouch(SeekBar seekBar) {

      }

      @Override
      public void onStopTrackingTouch(SeekBar seekBar) {
        metros(seekBar.getProgress());
        // Esse metodo serve para quando a pessoa arrasta e soltar em determinado lugar ele ser executado
      }
    });

    switch1.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
      @Override
      public void onCheckedChanged(CompoundButton compoundButton, boolean b) {
        Vibrator vibrator = (Vibrator) getSystemService(VIBRATOR_SERVICE); // Instancio a o service VIBRATOR_SERVICE para poder utilizar a vibração do celular
        if(b){
          mediaPlayer = mediaPlayer.create(getApplicationContext(), R.raw.ligado); // Instancio a música ligar
          mediaPlayer.start(); //faço a música ligar ser iniciada
          textStatus.setText("LIGADO"); // mudo o text para Ligado
          seekBar.setEnabled(true); // deixo ativado para poder mover o seekbar
          vibrator.vibrate(500);
          vibrator.vibrate(500);
          // faço ele vibrar duas vezes para dizer que ele está ligado
          recebe(b);
        }else{
          mediaPlayer = mediaPlayer.create(getApplicationContext(), R.raw.desligado); // Instancio a música desligado
          mediaPlayer.start(); //faço a música desligado ser iniciada
          textStatus.setText("DESLIGADO"); // mudo o text para desligado
          seekBar.setEnabled(false); // deixo desativado para poder mover o seekbar
          vibrator.vibrate(500);
          // faço ele vibrar uma vezes para dizer que ele está desligado
          recebe(b);
        }
      }
    });



    trackingOverlay = (OverlayView) findViewById(R.id.tracking_overlay);
    trackingOverlay.addCallback(
        new DrawCallback() {
          @Override
          public void drawCallback(final Canvas canvas) {
            tracker.draw(canvas);
            if (isDebug()) {
              tracker.drawDebug(canvas);
            }
          }
        });

    tracker.setFrameConfiguration(previewWidth, previewHeight, sensorOrientation);
  }


  public void metros(int i){
    Vibrator vibrator = (Vibrator) getSystemService(VIBRATOR_SERVICE); //Instancio o service VIBRATOR_SERVICE para poder manuzear a vibração do celular
    // Neste metodo eu passo em qual número do seekbar ele está dizendo qual a preferencia de metros que o usuário quer, de acordo com este número
    // eu entro em algum dos if abaixo e rodo uma música dizendo em quantos metros parou e vibro em uma determina frequencia avisando também em
    // quantos metros parou
    if(i == 0){
      mediaPlayer = mediaPlayer.create(getApplicationContext(), R.raw.musica1);
      mediaPlayer.start();
      textMetros.setText("1 metro");
      vibrator.vibrate(50);
    }else if(i == 1){
      mediaPlayer = mediaPlayer.create(getApplicationContext(), R.raw.musica2);
      mediaPlayer.start();
      textMetros.setText("2 metros");
      vibrator.vibrate(100);

    }else if(i == 2){
      mediaPlayer = mediaPlayer.create(getApplicationContext(), R.raw.musica3);
      mediaPlayer.start();
      textMetros.setText("3 metros");
      vibrator.vibrate(150);

    }else if(i == 3){
      mediaPlayer = mediaPlayer.create(getApplicationContext(), R.raw.musica4);
      mediaPlayer.start();
      textMetros.setText("4 metros");
      vibrator.vibrate(200);

    }else if(i == 4){
      mediaPlayer = mediaPlayer.create(getApplicationContext(), R.raw.musica5);
      mediaPlayer.start();
      textMetros.setText("5 metros");
      vibrator.vibrate(250);
    }
  }

  public void salvarPreferencia(){
    //metodo desenvolvido para salvar a preferencia do usuário assim caso ele queira sair do aplicativo e voltar ele não precise ficar reconfigurando toda hora
    SharedPreferences preferences = getSharedPreferences(METROS_PREFERENCIA, 0); // Instancio o metodo SharedPreferences
    SharedPreferences.Editor editor = preferences.edit(); //Edito o metodo

    int metros = seekBar.getProgress(); //Pego em qual número parou o seekBar
    editor.putInt("metros", metros); //Cria uma variavel com o nome metros e salvo em quantos metros parou
    editor.commit(); // faço o commit para poder salvar no aplicativo este dado
  }

  public void recuperarPreferencia(){
    SharedPreferences preferences = getSharedPreferences(METROS_PREFERENCIA, 0);// Instancio o metodo SharedPreferences
    if(preferences.contains("metros")){ // verifico se tem a variavel metros
      int metros = preferences.getInt("metros", 5); // caso tenha pego qual valor estava salvo e atribuo a outra variavel e caso não tiver eu passo o valor 5
      seekBar.setProgress(metros); // passo o valor que foi salvo pelo usuário
      metros(metros);
    }else{
      seekBar.setProgress(5); // caso não tenha sido configurado ainda a preferencia ele passa o valor padrão que seria 5 metros
    }
  }

  @Override
  public synchronized void onStart() {
    super.onStart();
  }

  @Override
  public void onDestroy() {
    super.onDestroy();
    salvarPreferencia();
  }

  @Override
  public void onPause() {
    super.onPause();
    salvarPreferencia();
  }

  public void recebe(boolean b){
    this.b = b;
  }

  public void vibrar(double distancia){
    Vibrator vibrator = (Vibrator) getSystemService(VIBRATOR_SERVICE); //Instancio o service VIBRATOR_SERVICE para poder manuzear a vibração do celular
    if(this.b){ // desligado

    }else{ // ligado
      if(distancia <= seekBar.getProgress()){ // verifico se a distancia é menor ou igual a preferencia do usuário, e dependendo da distancia entre o objeto e a pessoa dispara uma vibração
        if(distancia > 4 && distancia <= 5){
          vibrator.vibrate(1000);
        }
        else if( distancia > 3 && distancia <= 4){
          vibrator.vibrate(800);
        }
        else if( distancia > 2 && distancia <= 3){
          vibrator.vibrate(600);
        }
        else if( distancia > 1 && distancia <= 2){
          vibrator.vibrate(400);
        }
        else if( distancia == 1){
          vibrator.vibrate(200);
        }
      }
    }
  }




  @Override
  protected void processImage() {
    ++timestamp;
    final long currTimestamp = timestamp;
    trackingOverlay.postInvalidate();

    // No mutex needed as this method is not reentrant.
    if (computingDetection) {
      readyForNextImage();
      return;
    }
    computingDetection = true;
    LOGGER.i("Preparing image " + currTimestamp + " for detection in bg thread.");

    rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);

    readyForNextImage();
    final Canvas canvas = new Canvas(croppedBitmap);
    canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);
    // For examining the actual TF input.
    if (SAVE_PREVIEW_BITMAP) {
      ImageUtils.saveBitmap(croppedBitmap);
    }

    runInBackground(
        new Runnable() {
          @Override
          public void run() {
            LOGGER.i("Running detection on image " + currTimestamp);
            final long startTime = SystemClock.uptimeMillis();
            final List<Classifier.Recognition> results = detector.recognizeImage(croppedBitmap);
            lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;

            cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
            final Canvas canvas = new Canvas(cropCopyBitmap);
            final Paint paint = new Paint();
            paint.setColor(Color.RED);
            paint.setStyle(Style.STROKE);
            paint.setStrokeWidth(2.0f);

            float minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
            switch (MODE) {
              case TF_OD_API:
                minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
                break;
            }

            final List<Classifier.Recognition> mappedRecognitions =
                new LinkedList<Classifier.Recognition>();

            for (final Classifier.Recognition result : results) {
              final RectF location = result.getLocation();
              if (location != null && result.getConfidence() >= minimumConfidence) {
                canvas.drawRect(location, paint);

                cropToFrameTransform.mapRect(location);

                result.setLocation(location);
                mappedRecognitions.add(result);

                float topDiference = result.getLocation().top;
                float bottomDiference= result.getLocation().bottom;

                //calcula a altura do frame gerado sob o objeto detectado
                float frameVerticalTotal = 480 - (bottomDiference + topDiference);

                float rightDiference = result.getLocation().right;
                float leftDiference = result.getLocation().left;

                //calcula a largura do frame gerado sob o objeto detectado
                float frameHorizontalTotal = 640 - (rightDiference+leftDiference);

                //atribui em double distance, a distância focal(117°) * a largura do frame/ pela largura do bitmap
                double distance = (318.174367 * frameHorizontalTotal/640);



                if(result.getTitle().equals("person")) {
                  Toast.makeText(DetectorActivity.this,"objeto detectado: ", Toast.LENGTH_SHORT).show();
                  //transforma a distancia em metros
                  distance = distance / 100;

                  //verifica se a distancia calculada eh < 0.. se for, multiplica por -1
                  if(distance<0)
                    distance *= -1;

                  //gera um Toast mostrando a distância calculada do objeto detectado
                  Toast.makeText(DetectorActivity.this,"Distance: " + distance, Toast.LENGTH_SHORT).show();

                  vibrar(distance);

                }
              }
            }

            tracker.trackResults(mappedRecognitions, currTimestamp);
            trackingOverlay.postInvalidate();

            computingDetection = false;

            runOnUiThread(
                new Runnable() {
                  @Override
                  public void run() {
                    showFrameInfo(previewWidth + "x" + previewHeight);
                    showCropInfo(cropCopyBitmap.getWidth() + "x" + cropCopyBitmap.getHeight());
                    showInference(lastProcessingTimeMs + "ms");
                  }
                });
          }
        });
  }

  @Override
  protected int getLayoutId() {
    return R.layout.tfe_od_camera_connection_fragment_tracking;
  }

  @Override
  protected Size getDesiredPreviewFrameSize() {
    return DESIRED_PREVIEW_SIZE;
  }

  // Which detection model to use: by default uses Tensorflow Object Detection API frozen
  // checkpoints.
  private enum DetectorMode {
    TF_OD_API;
  }

  @Override
  protected void setUseNNAPI(final boolean isChecked) {
    runInBackground(() -> detector.setUseNNAPI(isChecked));
  }

  @Override
  protected void setNumThreads(final int numThreads) {
    runInBackground(() -> detector.setNumThreads(numThreads));
  }
}
