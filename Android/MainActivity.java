package com.example.language_translator;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.speech.RecognitionListener;
import android.speech.RecognizerIntent;
import android.speech.SpeechRecognizer;
import android.speech.tts.TextToSpeech;
import android.view.MotionEvent;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.EditText;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import org.json.JSONObject;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Locale;

import okhttp3.Call;
import okhttp3.Callback;
import okhttp3.MediaType;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

public class MainActivity extends AppCompatActivity {

    // ============= UI ELEMENTS =============
    private EditText tvSpokenText;
    private TextView tvTranslatedText;
    private TextView tvStatus;
    private Spinner spinnerSourceLang;
    private Spinner spinnerTargetLang;
    private Button btnMic;
    private Button btnTranslate;
    private Button btnSwapLang;

    // ============= SPEECH RECOGNITION =============
    private SpeechRecognizer speechRecognizer;
    private Intent speechIntent;

    // ============= TTS =============
    private TextToSpeech tts;

    // ============= TRANSLATION SERVER =============
    private static final String SERVER_URL = "http://192.168.191.37:5000/translate";
    private OkHttpClient httpClient = new OkHttpClient();

    // ============= LANGUAGE DIRECTION =============
    // Languages list — add more here when new models are trained
    private final String[] LANGUAGES = {"English", "Hindi", "Tamil", "Bengali"};
    private String sourceLang = "English";
    private String targetLang = "Hindi";

    // ============= PERMISSIONS =============
    private static final int PERMISSION_REQUEST_CODE = 1;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Link UI elements
        tvSpokenText      = findViewById(R.id.tvSpokenText);
        tvTranslatedText  = findViewById(R.id.tvTranslatedText);
        tvStatus          = findViewById(R.id.tvStatus);
        spinnerSourceLang = findViewById(R.id.spinnerSourceLang);
        spinnerTargetLang = findViewById(R.id.spinnerTargetLang);
        btnMic            = findViewById(R.id.btnMic);
        btnTranslate      = findViewById(R.id.btnTranslate);
        btnSwapLang       = findViewById(R.id.btnSwapLang);

        // Initialize TTS
        tts = new TextToSpeech(this, status -> {
            if (status == TextToSpeech.SUCCESS) {
                tvStatus.setText("● Ready");
            } else {
                tvStatus.setText("TTS failed to initialize");
            }
        });

        // Setup language spinners
        setupSpinners();

        // Request microphone permission
        requestPermissions();

        // Setup speech recognizer
        setupSpeechRecognizer();

        // Mic button — hold to speak
        btnMic.setOnTouchListener((v, event) -> {
            if (event.getAction() == MotionEvent.ACTION_DOWN) {
                startListening();
            } else if (event.getAction() == MotionEvent.ACTION_UP) {
                stopListening();
            }
            return true;
        });

        // Translate button
        btnTranslate.setOnClickListener(v -> {
            String spokenText = tvSpokenText.getText().toString().trim();
            if (spokenText.isEmpty()) {
                Toast.makeText(this, "Please speak or type something first",
                        Toast.LENGTH_SHORT).show();
                return;
            }
            sendToServer(spokenText);
        });

        // Swap language button
        btnSwapLang.setOnClickListener(v -> swapLanguage());
    }

    // ============= SPINNER SETUP =============
    private void setupSpinners() {
        ArrayAdapter<String> adapter = new ArrayAdapter<String>(
                this,
                android.R.layout.simple_spinner_item,
                LANGUAGES
        ) {
            @Override
            public View getView(int position, View convertView, android.view.ViewGroup parent) {
                TextView tv = (TextView) super.getView(position, convertView, parent);
                tv.setTextColor(0xFF1A3276);
                tv.setTextSize(16);
                tv.setTypeface(null, android.graphics.Typeface.BOLD);
                return tv;
            }

            @Override
            public View getDropDownView(int position, View convertView, android.view.ViewGroup parent) {
                TextView tv = (TextView) super.getDropDownView(position, convertView, parent);
                tv.setTextColor(0xFF1A3276);
                tv.setTextSize(16);
                tv.setPadding(20, 20, 20, 20);
                return tv;
            }
        };

        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);

        spinnerSourceLang.setAdapter(adapter);
        spinnerTargetLang.setAdapter(adapter);

        // Default: English → Hindi
        spinnerSourceLang.setSelection(0);
        spinnerTargetLang.setSelection(1);

        // Source language change listener
        spinnerSourceLang.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                sourceLang = LANGUAGES[position];
            }
            @Override
            public void onNothingSelected(AdapterView<?> parent) {}
        });

        // Target language change listener
        spinnerTargetLang.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                targetLang = LANGUAGES[position];
            }
            @Override
            public void onNothingSelected(AdapterView<?> parent) {}
        });
    }

    // ============= GET DIRECTION STRING =============
    private String getDirection() {
        // Maps language pair to server direction string
        if (sourceLang.equals("English") && targetLang.equals("Hindi")) return "en-hi";
        if (sourceLang.equals("Hindi") && targetLang.equals("English")) return "hi-en";
        // Default fallback
        return "en-hi";
    }

    // ============= GET STT LANGUAGE CODE =============
    private String getSttLanguage() {
        if (sourceLang.equals("Hindi")) return "hi-IN";
        return "en-US";
    }

    // ============= GET TTS LOCALE =============
    private Locale getTtsLocale() {
        if (targetLang.equals("Hindi")) return new Locale("hi", "IN");
        return Locale.ENGLISH;
    }

    // ============= PERMISSIONS =============
    private void requestPermissions() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.RECORD_AUDIO},
                    PERMISSION_REQUEST_CODE);
        }
    }

    // ============= SPEECH RECOGNIZER SETUP =============
    private void setupSpeechRecognizer() {
        speechRecognizer = SpeechRecognizer.createSpeechRecognizer(this);

        speechIntent = new Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH);
        speechIntent.putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL,
                RecognizerIntent.LANGUAGE_MODEL_FREE_FORM);
        speechIntent.putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true);

        speechRecognizer.setRecognitionListener(new RecognitionListener() {

            @Override
            public void onReadyForSpeech(Bundle params) {
                tvStatus.setText("● Listening...");
                btnMic.setText("🔴  LISTENING...");
            }

            @Override
            public void onPartialResults(Bundle partialResults) {
                ArrayList<String> partial = partialResults
                        .getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION);
                if (partial != null && !partial.isEmpty()) {
                    tvSpokenText.setText(partial.get(0));
                }
            }

            @Override
            public void onResults(Bundle results) {
                ArrayList<String> matches = results
                        .getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION);
                if (matches != null && !matches.isEmpty()) {
                    tvSpokenText.setText(matches.get(0));
                    tvStatus.setText("● Ready — press Translate");
                } else {
                    tvStatus.setText("● Could not hear clearly. Try again.");
                }
                btnMic.setText("🎤  HOLD TO SPEAK");
            }

            @Override
            public void onError(int error) {
                tvStatus.setText("● Mic error: " + getMicError(error));
                btnMic.setText("🎤  HOLD TO SPEAK");
            }

            @Override public void onBeginningOfSpeech() {}
            @Override public void onRmsChanged(float rmsdB) {}
            @Override public void onBufferReceived(byte[] buffer) {}
            @Override public void onEndOfSpeech() {}
            @Override public void onEvent(int eventType, Bundle params) {}
        });
    }

    // ============= START / STOP LISTENING =============
    private void startListening() {
        String lang = getSttLanguage();
        speechIntent.putExtra(RecognizerIntent.EXTRA_LANGUAGE, lang);
        speechIntent.putExtra(RecognizerIntent.EXTRA_LANGUAGE_PREFERENCE, lang);

        tvTranslatedText.setText("Translation will appear here...");
        tvStatus.setText("● Listening...");
        speechRecognizer.startListening(speechIntent);
    }

    private void stopListening() {
        speechRecognizer.stopListening();
        btnMic.setText("🎤  HOLD TO SPEAK");
    }

    // ============= SWAP LANGUAGE =============
    private void swapLanguage() {
        // Swap spinner selections
        int srcPos = spinnerSourceLang.getSelectedItemPosition();
        int tgtPos = spinnerTargetLang.getSelectedItemPosition();
        spinnerSourceLang.setSelection(tgtPos);
        spinnerTargetLang.setSelection(srcPos);

        // Clear boxes
        tvSpokenText.setText("");
        tvTranslatedText.setText("Translation will appear here...");
        tvStatus.setText("● Ready");
    }

    // ============= SEND TO SERVER =============
    private void sendToServer(String text) {
        tvStatus.setText("● Translating...");
        btnTranslate.setEnabled(false);

        try {
            JSONObject json = new JSONObject();
            json.put("text", text);
            json.put("direction", getDirection());

            RequestBody body = RequestBody.create(
                    json.toString(),
                    MediaType.parse("application/json")
            );

            Request request = new Request.Builder()
                    .url(SERVER_URL)
                    .post(body)
                    .build();

            httpClient.newCall(request).enqueue(new Callback() {

                @Override
                public void onResponse(Call call, Response response) throws IOException {
                    String responseBody = response.body().string();
                    runOnUiThread(() -> {
                        try {
                            JSONObject jsonResponse = new JSONObject(responseBody);
                            String translated = jsonResponse.getString("translation");
                            tvTranslatedText.setText(translated);
                            tvStatus.setText("● Done ✓");
                            speakTranslation(translated);
                        } catch (Exception e) {
                            tvStatus.setText("● Error reading response");
                        }
                        btnTranslate.setEnabled(true);
                    });
                }

                @Override
                public void onFailure(Call call, IOException e) {
                    runOnUiThread(() -> {
                        tvStatus.setText("● Cannot reach server — check WiFi");
                        btnTranslate.setEnabled(true);
                    });
                }
            });

        } catch (Exception e) {
            tvStatus.setText("● Request error: " + e.getMessage());
            btnTranslate.setEnabled(true);
        }
    }

    // ============= TTS =============
    private void speakTranslation(String text) {
        Locale locale = getTtsLocale();
        int result = tts.setLanguage(locale);

        if (result == TextToSpeech.LANG_MISSING_DATA ||
                result == TextToSpeech.LANG_NOT_SUPPORTED) {
            tvStatus.setText("● TTS language not supported — install from Settings");
            return;
        }

        tts.speak(text, TextToSpeech.QUEUE_FLUSH, null, "tts_id");
    }

    // ============= MIC ERROR HELPER =============
    private String getMicError(int errorCode) {
        switch (errorCode) {
            case SpeechRecognizer.ERROR_AUDIO:          return "Audio error";
            case SpeechRecognizer.ERROR_NETWORK:        return "No internet";
            case SpeechRecognizer.ERROR_NO_MATCH:       return "No speech detected";
            case SpeechRecognizer.ERROR_SPEECH_TIMEOUT: return "Timeout — speak faster";
            default:                                    return "Unknown error " + errorCode;
        }
    }

    // ============= CLEANUP =============
    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (speechRecognizer != null) {
            speechRecognizer.destroy();
        }
        if (tts != null) {
            tts.stop();
            tts.shutdown();
        }
    }
}