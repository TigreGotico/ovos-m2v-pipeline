# Model Evaluation Metrics Summary
## multilingual - Model: minishlab/M2V_multilingual_output
### Accuracy: 0.9923011120615911
### F1 Score: 0.9918944223607419
### Classification Report:
```
                                                                        precision    recall  f1-score   support

                              ovos-skill-alerts.openvoiceos:deletelist       0.00      0.00      0.00         0
                       ovos-skill-alerts.openvoiceos:deletelistentries       0.00      0.00      0.00         1
                              ovos-skill-alerts.openvoiceos:listalerts       1.00      1.00      1.00         1
                    ovos-skill-alerts.openvoiceos:missed_alerts.intent       1.00      1.00      1.00       636
                         ovos-skill-alerts.openvoiceos:reschedulealert       1.00      1.00      1.00         1
         ovos-skill-audio-recording.openvoiceos:start_recording.intent       0.99      1.00      1.00       100
             ovos-skill-boot-finished.openvoiceos:are_you_ready.intent       1.00      1.00      1.00        25
ovos-skill-boot-finished.openvoiceos:disable_ready_notification.intent       1.00      1.00      1.00      1103
 ovos-skill-boot-finished.openvoiceos:enable_ready_notification.intent       1.00      1.00      1.00      1302
                      ovos-skill-camera.openvoiceos:have_camera.intent       1.00      1.00      1.00         2
                     ovos-skill-camera.openvoiceos:take_picture.intent       1.00      1.00      1.00         1
                    ovos-skill-confucius-quotes.openvoiceos:who.intent       1.00      1.00      1.00         1
           ovos-skill-date-time.openvoiceos:date.future.weekend.intent       1.00      1.00      1.00        94
             ovos-skill-date-time.openvoiceos:date.last.weekend.intent       1.00      1.00      1.00       308
                    ovos-skill-date-time.openvoiceos:handle_query_time       0.00      0.00      0.00         1
                ovos-skill-date-time.openvoiceos:what.day.is.it.intent       1.00      1.00      1.00         6
              ovos-skill-date-time.openvoiceos:what.month.is.it.intent       1.00      1.00      1.00         7
               ovos-skill-date-time.openvoiceos:what.time.is.it.intent       1.00      1.00      1.00       253
          ovos-skill-date-time.openvoiceos:what.time.will.it.be.intent       1.00      1.00      1.00       179
            ovos-skill-date-time.openvoiceos:what.weekday.is.it.intent       0.80      1.00      0.89         4
               ovos-skill-date-time.openvoiceos:what.year.is.it.intent       1.00      1.00      1.00         7
       ovos-skill-days-in-history.openvoiceos:deaths_in_history.intent       1.00      1.00      1.00         1
        ovos-skill-days-in-history.openvoiceos:today_in_history.intent       1.00      1.00      1.00       154
                        ovos-skill-ddg.openvoiceos:age_at_death.intent       1.00      0.80      0.89         5
                           ovos-skill-ddg.openvoiceos:birthdate.intent       0.80      1.00      0.89         4
                                ovos-skill-ddg.openvoiceos:born.intent       1.00      0.83      0.91         6
                            ovos-skill-ddg.openvoiceos:children.intent       1.00      1.00      1.00         3
                                ovos-skill-ddg.openvoiceos:died.intent       0.89      1.00      0.94         8
                           ovos-skill-ddg.openvoiceos:education.intent       1.00      1.00      1.00         4
                           ovos-skill-ddg.openvoiceos:known_for.intent       1.00      1.00      1.00         3
                       ovos-skill-ddg.openvoiceos:resting_place.intent       1.00      0.33      0.50         3
                         ovos-skill-ddg.openvoiceos:search_duck.intent       1.00      1.00      1.00        14
             ovos-skill-diagnostics.openvoiceos:query_cpu_usage.intent       1.00      1.00      1.00        16
           ovos-skill-diagnostics.openvoiceos:query_extra_langs.intent       0.88      1.00      0.94        29
                   ovos-skill-diagnostics.openvoiceos:query_gpu.intent       1.00      1.00      1.00         9
        ovos-skill-diagnostics.openvoiceos:query_kernel_version.intent       1.00      1.00      1.00         2
                 ovos-skill-diagnostics.openvoiceos:query_langs.intent       1.00      0.96      0.98        95
          ovos-skill-diagnostics.openvoiceos:query_memory_usage.intent       1.00      1.00      1.00        10
         ovos-skill-diagnostics.openvoiceos:query_ovos_location.intent       0.96      1.00      0.98        26
          ovos-skill-diagnostics.openvoiceos:query_primary_lang.intent       1.00      1.00      1.00        49
             ovos-skill-diagnostics.openvoiceos:query_user_lang.intent       1.00      0.95      0.98        22
         ovos-skill-diagnostics.openvoiceos:query_user_location.intent       1.00      1.00      1.00        12
               ovos-skill-dictation.openvoiceos:start_dictation.intent       1.00      1.00      1.00       104
                ovos-skill-dictation.openvoiceos:stop_dictation.intent       1.00      0.98      0.99        52
                   ovos-skill-hello-world.openvoiceos:greetings.intent       0.82      0.86      0.84        21
                    ovos-skill-icanhazdadjokes.openvoiceos:joke.intent       0.50      0.50      0.50         2
             ovos-skill-icanhazdadjokes.openvoiceos:search_joke.intent       1.00      1.00      1.00        12
                            ovos-skill-ip.openvoiceos:what.ssid.intent       1.00      1.00      1.00        65
                      ovos-skill-iss-location.openvoiceos:about.intent       0.99      1.00      1.00       118
                   ovos-skill-iss-location.openvoiceos:when_iss.intent       1.00      1.00      1.00       188
                  ovos-skill-iss-location.openvoiceos:where_iss.intent       1.00      1.00      1.00        51
                           ovos-skill-laugh.openvoiceos:haunted.intent       1.00      1.00      1.00         1
                             ovos-skill-laugh.openvoiceos:laugh.intent       0.80      0.89      0.84         9
                       ovos-skill-laugh.openvoiceos:randomlaugh.intent       0.89      0.89      0.89         9
           ovos-skill-moviemaster.openvoiceos:movie.description.intent       1.00      1.00      1.00         1
                         ovos-skill-naptime.openvoiceos:naptime.intent       0.88      0.96      0.92        23
                        ovos-skill-news.openvoiceos:global_news.intent       1.00      1.00      1.00        35
                               ovos-skill-news.openvoiceos:news.intent       1.00      0.97      0.99       112
                  ovos-skill-parrot.openvoiceos:did.you.hear.me.intent       0.40      1.00      0.57         2
                       ovos-skill-parrot.openvoiceos:repeat.stt.intent       0.86      0.75      0.80         8
                       ovos-skill-parrot.openvoiceos:repeat.tts.intent       0.84      0.91      0.88        23
                            ovos-skill-parrot.openvoiceos:speak.intent       1.00      0.50      0.67         2
                     ovos-skill-parrot.openvoiceos:start_parrot.intent       1.00      0.89      0.94         9
                      ovos-skill-parrot.openvoiceos:stop_parrot.intent       1.00      0.89      0.94        19
    ovos-skill-personal.openvoiceos.openvoiceos:whenwereyouborn.intent       0.00      0.00      0.00         1
                     ovos-skill-personal.openvoiceos:whatareyou.intent       1.00      1.00      1.00         5
                ovos-skill-personal.openvoiceos:whenwereyouborn.intent       0.92      1.00      0.96        12
               ovos-skill-personal.openvoiceos:wherewereyouborn.intent       1.00      0.88      0.93         8
                      ovos-skill-personal.openvoiceos:whoareyou.intent       1.00      0.75      0.86         4
                     ovos-skill-personal.openvoiceos:whomadeyou.intent       1.00      1.00      1.00        19
           ovos-skill-randomness.openvoiceos:roll-multiple-dice.intent       1.00      1.00      1.00         3
              ovos-skill-randomness.openvoiceos:roll-single-die.intent       1.00      1.00      1.00         1
              ovos-skill-screenshot.openvoiceos:take.screenshot.intent       1.00      1.00      1.00         1
                         ovos-skill-volume.openvoiceos:increase_volume       1.00      0.50      0.67         2
                   ovos-skill-volume.openvoiceos:volume.default.intent       1.00      1.00      1.00        17
                      ovos-skill-volume.openvoiceos:volume.high.intent       0.86      0.86      0.86         7
                       ovos-skill-volume.openvoiceos:volume.low.intent       1.00      1.00      1.00         9
                       ovos-skill-volume.openvoiceos:volume.max.intent       1.00      1.00      1.00        11
                             ovos-skill-volume.openvoiceos:volume.mute       0.00      0.00      0.00         1
                      ovos-skill-volume.openvoiceos:volume.mute.intent       0.75      0.75      0.75         4
               ovos-skill-volume.openvoiceos:volume.mute.toggle.intent       1.00      0.80      0.89         5
                    ovos-skill-volume.openvoiceos:volume.unmute.intent       0.71      0.71      0.71         7
                    ovos-skill-weather.openvoiceos:current_temperature       0.00      0.00      0.00         1
                      ovos-skill-weather.openvoiceos:daily_temperature       0.00      0.00      0.00         0
           ovos-skill-weather.openvoiceos:do-i-need-an-umbrella.intent       0.98      1.00      0.99        46
           ovos-skill-weather.openvoiceos:do.i.need.an.umbrella.intent       0.00      0.00      0.00         2
                           ovos-skill-weather.openvoiceos:week_weather       1.00      1.00      1.00         3
                         ovos-skill-wikihow.openvoiceos:wikihow.intent       1.00      1.00      1.00        17
                          ovos-skill-wikipedia.openvoiceos:wiki.intent       1.00      1.00      1.00        33
                  ovos-skill-wikipedia.openvoiceos:wikiroulette.intent       1.00      1.00      1.00        56
                    ovos-skill-wolfie.openvoiceos:search_wolfie.intent       1.00      1.00      1.00        34
                         ovos-skill-wordnet.openvoiceos:antonym.intent       1.00      1.00      1.00         8
                      ovos-skill-wordnet.openvoiceos:definition.intent       1.00      1.00      1.00         4
                         ovos-skill-wordnet.openvoiceos:holonym.intent       0.71      1.00      0.83         5
                        ovos-skill-wordnet.openvoiceos:hypernym.intent       1.00      1.00      1.00        14
                         ovos-skill-wordnet.openvoiceos:hyponym.intent       1.00      0.97      0.98        31
                           ovos-skill-wordnet.openvoiceos:lemma.intent       1.00      1.00      1.00         3
                  ovos-skill-wordnet.openvoiceos:search_wordnet.intent       1.00      1.00      1.00        13
                                     ovos.common_play.play_search:play       1.00      1.00      1.00        85

                                                              accuracy                           0.99      5845
                                                             macro avg       0.88      0.87      0.87      5845
                                                          weighted avg       0.99      0.99      0.99      5845

```

