# Model Evaluation Metrics Summary
## en - Model: minishlab/potion-base-4M
### Accuracy: 0.9059233449477352
### F1 Score: 0.8933463841905207
### Classification Report:
```
                                                                        precision    recall  f1-score   support

                                                              ocp:play       1.00      1.00      1.00        83
                        ovos-skill-alerts.openvoiceos:ChangeProperties       0.00      0.00      0.00         0
                             ovos-skill-alerts.openvoiceos:CreateEvent       0.00      0.00      0.00         1
                          ovos-skill-alerts.openvoiceos:CreateReminder       0.00      0.00      0.00         1
                              ovos-skill-alerts.openvoiceos:ListAlerts       0.00      0.00      0.00         0
                         ovos-skill-alerts.openvoiceos:RescheduleAlert       0.00      0.00      0.00         1
                             ovos-skill-alerts.openvoiceos:TimerStatus       0.00      0.00      0.00         1
                    ovos-skill-alerts.openvoiceos:missed_alerts.intent       0.95      1.00      0.98        21
         ovos-skill-audio-recording.openvoiceos:start_recording.intent       0.78      1.00      0.88         7
             ovos-skill-boot-finished.openvoiceos:are_you_ready.intent       1.00      1.00      1.00         4
ovos-skill-boot-finished.openvoiceos:disable_ready_notification.intent       1.00      1.00      1.00        10
 ovos-skill-boot-finished.openvoiceos:enable_ready_notification.intent       1.00      1.00      1.00         8
           ovos-skill-date-time.openvoiceos:date.future.weekend.intent       0.88      0.78      0.82         9
             ovos-skill-date-time.openvoiceos:date.last.weekend.intent       0.67      0.80      0.73         5
                    ovos-skill-date-time.openvoiceos:handle_query_time       0.00      0.00      0.00         2
              ovos-skill-date-time.openvoiceos:what.month.is.it.intent       1.00      1.00      1.00         1
               ovos-skill-date-time.openvoiceos:what.time.is.it.intent       0.78      1.00      0.88         7
          ovos-skill-date-time.openvoiceos:what.time.will.it.be.intent       1.00      1.00      1.00         6
       ovos-skill-days-in-history.openvoiceos:births_in_history.intent       0.00      0.00      0.00         1
        ovos-skill-days-in-history.openvoiceos:today_in_history.intent       1.00      1.00      1.00         3
                        ovos-skill-ddg.openvoiceos:age_at_death.intent       1.00      1.00      1.00         1
                           ovos-skill-ddg.openvoiceos:education.intent       0.00      0.00      0.00         0
                       ovos-skill-ddg.openvoiceos:resting_place.intent       0.00      0.00      0.00         1
             ovos-skill-diagnostics.openvoiceos:query_cpu_usage.intent       1.00      1.00      1.00         1
                 ovos-skill-diagnostics.openvoiceos:query_langs.intent       1.00      1.00      1.00         3
         ovos-skill-diagnostics.openvoiceos:query_ovos_location.intent       0.00      0.00      0.00         0
          ovos-skill-diagnostics.openvoiceos:query_primary_lang.intent       1.00      1.00      1.00         1
             ovos-skill-diagnostics.openvoiceos:query_user_lang.intent       1.00      1.00      1.00         2
               ovos-skill-dictation.openvoiceos:start_dictation.intent       0.95      0.90      0.93        21
                ovos-skill-dictation.openvoiceos:stop_dictation.intent       1.00      1.00      1.00         9
                   ovos-skill-hello-world.openvoiceos:Greetings.intent       0.50      0.50      0.50         2
                   ovos-skill-hello-world.openvoiceos:HowAreYou.intent       0.00      0.00      0.00         1
             ovos-skill-icanhazdadjokes.openvoiceos:search_joke.intent       1.00      1.00      1.00         1
                            ovos-skill-ip.openvoiceos:what.ssid.intent       1.00      1.00      1.00         2
                      ovos-skill-iss-location.openvoiceos:about.intent       0.00      0.00      0.00         1
                   ovos-skill-iss-location.openvoiceos:when_iss.intent       0.93      1.00      0.97        14
                  ovos-skill-iss-location.openvoiceos:where_iss.intent       0.67      0.67      0.67         3
                             ovos-skill-laugh.openvoiceos:Laugh.intent       1.00      1.00      1.00         1
                       ovos-skill-laugh.openvoiceos:RandomLaugh.intent       1.00      1.00      1.00         1
           ovos-skill-moviemaster.openvoiceos:movie.description.intent       0.00      0.00      0.00         0
           ovos-skill-moviemaster.openvoiceos:movie.information.intent       0.00      0.00      0.00         1
                        ovos-skill-news.openvoiceos:global_news.intent       1.00      1.00      1.00         5
                               ovos-skill-news.openvoiceos:news.intent       1.00      1.00      1.00         9
                  ovos-skill-parrot.openvoiceos:did.you.hear.me.intent       1.00      1.00      1.00         1
                       ovos-skill-parrot.openvoiceos:repeat.tts.intent       1.00      1.00      1.00         3
                            ovos-skill-parrot.openvoiceos:speak.intent       1.00      1.00      1.00         2
                     ovos-skill-parrot.openvoiceos:start_parrot.intent       1.00      1.00      1.00         1
                      ovos-skill-parrot.openvoiceos:stop_parrot.intent       1.00      1.00      1.00         1
                     ovos-skill-personal.OpenVoiceOS:WhatAreYou.intent       1.00      1.00      1.00         2
                ovos-skill-personal.OpenVoiceOS:WhenWereYouBorn.intent       0.00      0.00      0.00         0
               ovos-skill-personal.OpenVoiceOS:WhereWereYouBorn.intent       1.00      1.00      1.00         1
                     ovos-skill-personal.OpenVoiceOS:WhoMadeYou.intent       1.00      1.00      1.00         1
                  ovos-skill-randomness.openvoiceos:flip-a-coin.intent       0.00      0.00      0.00         1
               ovos-skill-randomness.openvoiceos:fortune-teller.intent       1.00      1.00      1.00         1
                ovos-skill-randomness.openvoiceos:make-a-choice.intent       1.00      1.00      1.00         1
           ovos-skill-randomness.openvoiceos:roll-multiple-dice.intent       0.50      1.00      0.67         1
                         ovos-skill-volume.openvoiceos:increase_volume       0.50      1.00      0.67         1
                             ovos-skill-volume.openvoiceos:less_volume       0.00      0.00      0.00         1
                   ovos-skill-volume.openvoiceos:volume.default.intent       1.00      1.00      1.00         1
                             ovos-skill-volume.openvoiceos:volume.mute       0.00      0.00      0.00         1
                      ovos-skill-volume.openvoiceos:volume.mute.intent       0.00      0.00      0.00         0
              ovos-skill-wallpapers.openvoiceos:wallpaper.about.intent       0.00      0.00      0.00         1
             ovos-skill-wallpapers.openvoiceos:wallpaper.random.intent       0.00      0.00      0.00         0
           ovos-skill-weather.openvoiceos:do-i-need-an-umbrella.intent       0.67      1.00      0.80         2
                       ovos-skill-weather.openvoiceos:high_temperature       1.00      1.00      1.00         1
                     ovos-skill-weather.openvoiceos:hourly_temperature       0.00      0.00      0.00         0
                               ovos-skill-weather.openvoiceos:is_clear       0.00      0.00      0.00         1
                              ovos-skill-weather.openvoiceos:is_cloudy       0.00      0.00      0.00         1
                        ovos-skill-weather.openvoiceos:low_temperature       0.00      0.00      0.00         1
                              ovos-skill-weather.openvoiceos:next_rain       0.00      0.00      0.00         1
                                ovos-skill-weather.openvoiceos:weather       0.50      1.00      0.67         1
                         ovos-skill-wikihow.openvoiceos:wikihow.intent       1.00      1.00      1.00         1
                          ovos-skill-wikipedia.openvoiceos:wiki.intent       1.00      1.00      1.00         3
                      ovos-skill-wordnet.openvoiceos:definition.intent       0.00      0.00      0.00         1
                           ovos-skill-wordnet.openvoiceos:lemma.intent       1.00      1.00      1.00         2
                  ovos-skill-wordnet.openvoiceos:search_wordnet.intent       0.00      0.00      0.00         0

                                                              accuracy                           0.91       287
                                                             macro avg       0.57      0.60      0.58       287
                                                          weighted avg       0.88      0.91      0.89       287

```

