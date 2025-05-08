# Model Evaluation Metrics Summary
## multilingual - Model: minishlab/M2V_multilingual_output
### Accuracy: 0.9916167664670659
### F1 Score: 0.9911375685802422
### Classification Report:
```
                                                                              precision    recall  f1-score   support

                                                                    ocp:play       1.00      1.00      1.00        76
                              ovos-skill-alerts.openvoiceos:ChangeProperties       0.00      0.00      0.00         0
                                   ovos-skill-alerts.openvoiceos:CreateEvent       0.00      0.00      0.00         1
                                       ovos-skill-alerts.openvoiceos:DAVSync       0.00      0.00      0.00         1
                          ovos-skill-alerts.openvoiceos:missed_alerts.intent       1.00      1.00      1.00       686
               ovos-skill-audio-recording.openvoiceos:start_recording.intent       0.99      1.00      0.99        76
                   ovos-skill-boot-finished.openvoiceos:are_you_ready.intent       0.96      1.00      0.98        25
      ovos-skill-boot-finished.openvoiceos:disable_ready_notification.intent       1.00      1.00      1.00      1055
       ovos-skill-boot-finished.openvoiceos:enable_ready_notification.intent       1.00      1.00      1.00      1283
                            ovos-skill-camera.openvoiceos:have_camera.intent       1.00      1.00      1.00         5
ovos-skill-color-picker.krisgesling.openvoiceos:request-color-by-name.intent       0.00      0.00      0.00         1
                          ovos-skill-confucius-quotes.openvoiceos:who.intent       1.00      1.00      1.00         3
                 ovos-skill-date-time.openvoiceos:date.future.weekend.intent       1.00      1.00      1.00       142
                   ovos-skill-date-time.openvoiceos:date.last.weekend.intent       1.00      1.00      1.00       304
                        ovos-skill-date-time.openvoiceos:handle_day_for_date       0.00      0.00      0.00         1
                      ovos-skill-date-time.openvoiceos:what.day.is.it.intent       0.80      1.00      0.89         4
                    ovos-skill-date-time.openvoiceos:what.month.is.it.intent       1.00      0.83      0.91         6
                     ovos-skill-date-time.openvoiceos:what.time.is.it.intent       1.00      0.99      0.99       279
                ovos-skill-date-time.openvoiceos:what.time.will.it.be.intent       1.00      1.00      1.00       189
                  ovos-skill-date-time.openvoiceos:what.weekday.is.it.intent       1.00      1.00      1.00         1
                     ovos-skill-date-time.openvoiceos:what.year.is.it.intent       1.00      1.00      1.00         7
             ovos-skill-days-in-history.openvoiceos:births_in_history.intent       1.00      1.00      1.00         1
              ovos-skill-days-in-history.openvoiceos:today_in_history.intent       1.00      1.00      1.00       157
                              ovos-skill-ddg.openvoiceos:age_at_death.intent       1.00      1.00      1.00         3
                                 ovos-skill-ddg.openvoiceos:birthdate.intent       1.00      1.00      1.00         8
                                      ovos-skill-ddg.openvoiceos:born.intent       1.00      0.88      0.93         8
                                      ovos-skill-ddg.openvoiceos:died.intent       1.00      1.00      1.00         3
                                 ovos-skill-ddg.openvoiceos:education.intent       1.00      1.00      1.00         1
                                 ovos-skill-ddg.openvoiceos:known_for.intent       1.00      1.00      1.00         1
                          ovos-skill-ddg.openvoiceos:official_website.intent       1.00      1.00      1.00         1
                             ovos-skill-ddg.openvoiceos:resting_place.intent       1.00      1.00      1.00         1
                               ovos-skill-ddg.openvoiceos:search_duck.intent       1.00      1.00      1.00        19
                   ovos-skill-diagnostics.openvoiceos:query_cpu_usage.intent       1.00      1.00      1.00         8
                 ovos-skill-diagnostics.openvoiceos:query_extra_langs.intent       0.95      0.88      0.91        24
                         ovos-skill-diagnostics.openvoiceos:query_gpu.intent       1.00      1.00      1.00         8
              ovos-skill-diagnostics.openvoiceos:query_kernel_version.intent       1.00      1.00      1.00         6
                       ovos-skill-diagnostics.openvoiceos:query_langs.intent       0.95      0.99      0.97        83
                ovos-skill-diagnostics.openvoiceos:query_memory_usage.intent       1.00      1.00      1.00         7
               ovos-skill-diagnostics.openvoiceos:query_ovos_location.intent       1.00      1.00      1.00        18
                ovos-skill-diagnostics.openvoiceos:query_primary_lang.intent       1.00      0.98      0.99        50
                   ovos-skill-diagnostics.openvoiceos:query_user_lang.intent       1.00      1.00      1.00        18
               ovos-skill-diagnostics.openvoiceos:query_user_location.intent       0.83      1.00      0.91         5
                     ovos-skill-dictation.openvoiceos:start_dictation.intent       1.00      0.99      1.00       120
                      ovos-skill-dictation.openvoiceos:stop_dictation.intent       1.00      1.00      1.00        29
                         ovos-skill-hello-world.openvoiceos:Greetings.intent       0.86      0.89      0.88        28
                          ovos-skill-icanhazdadjokes.openvoiceos:joke.intent       0.75      0.86      0.80         7
                   ovos-skill-icanhazdadjokes.openvoiceos:search_joke.intent       1.00      0.94      0.97        18
                                  ovos-skill-ip.openvoiceos:what.ssid.intent       1.00      1.00      1.00        56
                            ovos-skill-iss-location.openvoiceos:about.intent       0.99      1.00      0.99        83
                         ovos-skill-iss-location.openvoiceos:when_iss.intent       1.00      1.00      1.00       175
                        ovos-skill-iss-location.openvoiceos:where_iss.intent       1.00      1.00      1.00        41
                                   ovos-skill-laugh.openvoiceos:Laugh.intent       0.78      1.00      0.88        14
                             ovos-skill-laugh.openvoiceos:RandomLaugh.intent       1.00      0.82      0.90        11
                                 ovos-skill-laugh.openvoiceos:haunted.intent       1.00      1.00      1.00         2
                 ovos-skill-moviemaster.openvoiceos:movie.description.intent       0.00      0.00      0.00         0
                     ovos-skill-moviemaster.openvoiceos:movie.runtime.intent       0.00      0.00      0.00         1
                               ovos-skill-naptime.openvoiceos:naptime.intent       0.93      0.82      0.88        17
                              ovos-skill-news.openvoiceos:global_news.intent       1.00      0.95      0.98        42
                                     ovos-skill-news.openvoiceos:news.intent       0.98      1.00      0.99       110
                        ovos-skill-parrot.openvoiceos:did.you.hear.me.intent       1.00      0.89      0.94         9
                             ovos-skill-parrot.openvoiceos:repeat.stt.intent       0.75      0.75      0.75         4
                             ovos-skill-parrot.openvoiceos:repeat.tts.intent       0.91      1.00      0.95        31
                                  ovos-skill-parrot.openvoiceos:speak.intent       1.00      0.89      0.94         9
                           ovos-skill-parrot.openvoiceos:start_parrot.intent       0.82      0.90      0.86        10
                            ovos-skill-parrot.openvoiceos:stop_parrot.intent       0.92      1.00      0.96        24
         ovos-skill-personal.OpenVoiceOS.openvoiceos:WhereWereYouBorn.intent       0.00      0.00      0.00         1
                           ovos-skill-personal.OpenVoiceOS:WhatAreYou.intent       1.00      0.67      0.80         3
                      ovos-skill-personal.OpenVoiceOS:WhenWereYouBorn.intent       0.94      1.00      0.97        16
                     ovos-skill-personal.OpenVoiceOS:WhereWereYouBorn.intent       0.93      1.00      0.97        14
                            ovos-skill-personal.OpenVoiceOS:WhoAreYou.intent       0.75      1.00      0.86         3
                           ovos-skill-personal.OpenVoiceOS:WhoMadeYou.intent       1.00      1.00      1.00        22
                      ovos-skill-randomness.openvoiceos:make-a-choice.intent       0.00      0.00      0.00         1
                         ovos-skill-volume.openvoiceos:volume.default.intent       1.00      1.00      1.00        11
                            ovos-skill-volume.openvoiceos:volume.high.intent       1.00      1.00      1.00        14
                             ovos-skill-volume.openvoiceos:volume.low.intent       1.00      1.00      1.00         8
                             ovos-skill-volume.openvoiceos:volume.max.intent       1.00      1.00      1.00        11
                            ovos-skill-volume.openvoiceos:volume.mute.intent       0.92      0.92      0.92        12
                     ovos-skill-volume.openvoiceos:volume.mute.toggle.intent       0.80      0.80      0.80         5
                          ovos-skill-volume.openvoiceos:volume.unmute.intent       0.92      0.69      0.79        16
                    ovos-skill-wallpapers.openvoiceos:wallpaper.about.intent       0.00      0.00      0.00         1
                              ovos-skill-weather.openvoiceos:N_days_forecast       0.00      0.00      0.00         0
                          ovos-skill-weather.openvoiceos:current_temperature       0.00      0.00      0.00         0
                 ovos-skill-weather.openvoiceos:do-i-need-an-umbrella.intent       1.00      0.98      0.99        43
                             ovos-skill-weather.openvoiceos:high_temperature       0.00      0.00      0.00         0
                           ovos-skill-weather.openvoiceos:hourly_temperature       0.00      0.00      0.00         1
                                     ovos-skill-weather.openvoiceos:humidity       0.00      0.00      0.00         1
                                      ovos-skill-weather.openvoiceos:weather       1.00      0.67      0.80         3
                                 ovos-skill-weather.openvoiceos:week_weather       1.00      1.00      1.00         2
                               ovos-skill-wikihow.openvoiceos:wikihow.intent       1.00      1.00      1.00        27
                                ovos-skill-wikipedia.openvoiceos:wiki.intent       1.00      1.00      1.00        44
                        ovos-skill-wikipedia.openvoiceos:wikiroulette.intent       1.00      1.00      1.00        63
                          ovos-skill-wolfie.openvoiceos:search_wolfie.intent       1.00      1.00      1.00        23
                               ovos-skill-wordnet.openvoiceos:antonym.intent       1.00      1.00      1.00         6
                            ovos-skill-wordnet.openvoiceos:definition.intent       1.00      1.00      1.00         3
                               ovos-skill-wordnet.openvoiceos:holonym.intent       1.00      0.83      0.91         6
                              ovos-skill-wordnet.openvoiceos:hypernym.intent       1.00      1.00      1.00        15
                               ovos-skill-wordnet.openvoiceos:hyponym.intent       0.97      1.00      0.98        31
                                 ovos-skill-wordnet.openvoiceos:lemma.intent       1.00      1.00      1.00         8
                        ovos-skill-wordnet.openvoiceos:search_wordnet.intent       1.00      1.00      1.00        15

                                                                    accuracy                           0.99      5845
                                                                   macro avg       0.82      0.82      0.82      5845
                                                                weighted avg       0.99      0.99      0.99      5845

```

