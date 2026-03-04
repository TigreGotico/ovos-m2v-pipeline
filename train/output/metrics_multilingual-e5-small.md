# Model Evaluation – multilingual-e5-small

**Base model:** `/home/miro/PycharmProjects/ovos-m2v-pipeline/train/output/distilled/multilingual-e5-small`  
**Language filter:** all

## Dataset balance settings

| Setting | Value |
|---|---|
| Min samples per class | 10 |
| Max samples per class | 800 |
| Language filter | all |
| Training examples | 42062 |
| Test examples | 10516 |

## Model size

| Metric | Value |
|---|---|
| Disk size | 262.05 MB |
| Parameters | 64.50 M |

## Benchmark

| Metric | Value |
|---|---|
| Training time | 36.1s |
| Inference time (10516 samples) | 0.430s |
| Throughput | 24448 sps |

## Overall evaluation

| Metric | Value |
|---|---|
| Accuracy | 0.9496 |
| Weighted F1 | 0.9485 |

### Classification report

```
                                                                              precision    recall  f1-score   support

                                                   common_query:common_query       0.82      0.76      0.79       160
                                                    common_query:wiki.intent       0.60      0.50      0.55         6
                                                                    ocp:play       0.97      0.99      0.98       160
                               ovos-skill-alerts.openvoiceos:AddListSubitems       0.88      0.93      0.90        15
                                  ovos-skill-alerts.openvoiceos:CalendarList       0.83      1.00      0.91         5
                                   ovos-skill-alerts.openvoiceos:CancelAlert       1.00      0.82      0.90        11
                              ovos-skill-alerts.openvoiceos:ChangeProperties       0.60      0.43      0.50         7
                                   ovos-skill-alerts.openvoiceos:CreateAlarm       0.82      0.95      0.88        19
                                   ovos-skill-alerts.openvoiceos:CreateEvent       1.00      1.00      1.00         6
                                    ovos-skill-alerts.openvoiceos:CreateList       1.00      0.88      0.93         8
                                ovos-skill-alerts.openvoiceos:CreateOcpAlarm       0.93      0.91      0.92        47
                                ovos-skill-alerts.openvoiceos:CreateReminder       0.78      0.58      0.67        12
                                   ovos-skill-alerts.openvoiceos:CreateTimer       0.86      0.86      0.86         7
                                       ovos-skill-alerts.openvoiceos:DAVSync       1.00      0.60      0.75         5
                                    ovos-skill-alerts.openvoiceos:DeleteList       0.86      0.75      0.80         8
                             ovos-skill-alerts.openvoiceos:DeleteListEntries       0.82      0.90      0.86        10
                             ovos-skill-alerts.openvoiceos:DeleteTodoEntries       0.60      0.60      0.60        10
                                    ovos-skill-alerts.openvoiceos:ListAlerts       0.77      1.00      0.87        20
                              ovos-skill-alerts.openvoiceos:QueryListEntries       1.00      0.20      0.33         5
                                ovos-skill-alerts.openvoiceos:QueryListNames       0.67      0.80      0.73         5
                              ovos-skill-alerts.openvoiceos:QueryTodoEntries       0.67      0.40      0.50         5
                               ovos-skill-alerts.openvoiceos:RescheduleAlert       0.83      0.56      0.67         9
                                   ovos-skill-alerts.openvoiceos:TimerStatus       0.89      1.00      0.94        24
                          ovos-skill-alerts.openvoiceos:missed_alerts.intent       1.00      0.99      1.00       160
                    ovos-skill-application-launcher.openvoiceos:close.intent       0.78      0.88      0.82         8
                   ovos-skill-application-launcher.openvoiceos:launch.intent       0.89      0.80      0.84        10
               ovos-skill-audio-recording.openvoiceos:start_recording.intent       0.99      0.99      0.99       160
                   ovos-skill-boot-finished.openvoiceos:are_you_ready.intent       1.00      0.98      0.99       123
      ovos-skill-boot-finished.openvoiceos:disable_ready_notification.intent       0.97      0.99      0.98       160
       ovos-skill-boot-finished.openvoiceos:enable_ready_notification.intent       0.98      0.97      0.97       160
                            ovos-skill-camera.openvoiceos:have_camera.intent       1.00      0.92      0.96        12
                           ovos-skill-camera.openvoiceos:take_picture.intent       0.93      1.00      0.96        13
ovos-skill-color-picker.krisgesling.openvoiceos:request-color-by-name.intent       1.00      0.50      0.67         6
             ovos-skill-color-picker.krisgesling:request-color-by-hex.intent       1.00      1.00      1.00         2
            ovos-skill-color-picker.krisgesling:request-color-by-name.intent       1.00      1.00      1.00         6
                      ovos-skill-confucius-quotes.openvoiceos:ConfuciusBirth       0.75      0.43      0.55         7
                      ovos-skill-confucius-quotes.openvoiceos:ConfuciusDeath       0.56      0.71      0.62         7
                      ovos-skill-confucius-quotes.openvoiceos:ConfuciusQuote       1.00      1.00      1.00         8
                          ovos-skill-confucius-quotes.openvoiceos:who.intent       1.00      1.00      1.00         5
                              ovos-skill-count.openvoiceos:count_to_N.intent       0.99      1.00      1.00       160
                        ovos-skill-date-time.openvoiceos:current_date.intent       0.96      1.00      0.98       160
                 ovos-skill-date-time.openvoiceos:date.future.weekend.intent       1.00      0.99      0.99       160
                   ovos-skill-date-time.openvoiceos:date.last.weekend.intent       0.98      1.00      0.99       160
                        ovos-skill-date-time.openvoiceos:handle_day_for_date       0.86      0.95      0.90        19
                      ovos-skill-date-time.openvoiceos:next.leap.year.intent       1.00      1.00      1.00        77
                          ovos-skill-date-time.openvoiceos:time.until.intent       1.00      0.99      0.99        84
                    ovos-skill-date-time.openvoiceos:weekday.for.date.intent       0.99      0.99      0.99       160
                      ovos-skill-date-time.openvoiceos:what.day.is.it.intent       0.89      0.92      0.91        53
                    ovos-skill-date-time.openvoiceos:what.month.is.it.intent       0.99      0.99      0.99       160
                     ovos-skill-date-time.openvoiceos:what.time.is.it.intent       0.90      0.96      0.93       160
                ovos-skill-date-time.openvoiceos:what.time.will.it.be.intent       1.00      0.97      0.98       160
                  ovos-skill-date-time.openvoiceos:what.weekday.is.it.intent       0.92      0.92      0.92        71
                     ovos-skill-date-time.openvoiceos:what.year.is.it.intent       0.89      1.00      0.94        32
             ovos-skill-days-in-history.openvoiceos:births_in_history.intent       0.79      0.92      0.85        12
             ovos-skill-days-in-history.openvoiceos:deaths_in_history.intent       1.00      0.75      0.86         8
              ovos-skill-days-in-history.openvoiceos:today_in_history.intent       1.00      0.99      1.00       160
                              ovos-skill-ddg.openvoiceos:age_at_death.intent       1.00      0.89      0.94         9
                                 ovos-skill-ddg.openvoiceos:birthdate.intent       0.54      0.88      0.67         8
                                      ovos-skill-ddg.openvoiceos:born.intent       0.80      0.44      0.57         9
                                  ovos-skill-ddg.openvoiceos:children.intent       1.00      1.00      1.00         4
                                      ovos-skill-ddg.openvoiceos:died.intent       1.00      0.89      0.94         9
                                 ovos-skill-ddg.openvoiceos:education.intent       1.00      0.60      0.75         5
                                 ovos-skill-ddg.openvoiceos:known_for.intent       1.00      0.25      0.40         4
                          ovos-skill-ddg.openvoiceos:official_website.intent       1.00      0.50      0.67         2
                             ovos-skill-ddg.openvoiceos:resting_place.intent       1.00      1.00      1.00         4
                               ovos-skill-ddg.openvoiceos:search_duck.intent       1.00      1.00      1.00        40
                                       ovos-skill-ddg.openvoiceos:who.intent       0.50      0.50      0.50         2
                   ovos-skill-diagnostics.openvoiceos:query_cpu_usage.intent       0.96      1.00      0.98        25
                 ovos-skill-diagnostics.openvoiceos:query_extra_langs.intent       1.00      0.91      0.95        43
                         ovos-skill-diagnostics.openvoiceos:query_gpu.intent       1.00      1.00      1.00        14
              ovos-skill-diagnostics.openvoiceos:query_kernel_version.intent       1.00      1.00      1.00        14
                       ovos-skill-diagnostics.openvoiceos:query_langs.intent       0.94      0.99      0.97       160
                ovos-skill-diagnostics.openvoiceos:query_memory_usage.intent       1.00      1.00      1.00        26
               ovos-skill-diagnostics.openvoiceos:query_ovos_location.intent       0.96      0.94      0.95        48
                ovos-skill-diagnostics.openvoiceos:query_primary_lang.intent       0.99      0.99      0.99        93
                   ovos-skill-diagnostics.openvoiceos:query_user_lang.intent       0.97      1.00      0.99        37
               ovos-skill-diagnostics.openvoiceos:query_user_location.intent       0.93      0.96      0.94        26
                     ovos-skill-dictation.openvoiceos:start_dictation.intent       0.98      0.97      0.98       160
                      ovos-skill-dictation.openvoiceos:stop_dictation.intent       0.98      0.99      0.99       117
                   ovos-skill-fuster-quotes.openvoiceos:fuster_quotes.intent       1.00      1.00      1.00        12
                             ovos-skill-fuster-quotes.openvoiceos:who.intent       1.00      1.00      1.00         5
                         ovos-skill-hello-world.openvoiceos:Greetings.intent       0.83      0.79      0.81        97
                           ovos-skill-hello-world.openvoiceos:ThankYouIntent       1.00      0.71      0.83         7
                          ovos-skill-icanhazdadjokes.openvoiceos:joke.intent       0.98      0.97      0.98       160
                   ovos-skill-icanhazdadjokes.openvoiceos:search_joke.intent       1.00      0.97      0.98        29
                                          ovos-skill-ip.openvoiceos:IPIntent       1.00      1.00      1.00         9
                                  ovos-skill-ip.openvoiceos:what.ssid.intent       1.00      1.00      1.00        96
                         ovos-skill-iss-location.openvoiceos:NumberISSIntent       1.00      0.67      0.80         6
                            ovos-skill-iss-location.openvoiceos:WhoISSIntent       1.00      1.00      1.00        10
                         ovos-skill-iss-location.openvoiceos:when_iss.intent       0.99      1.00      1.00       160
                        ovos-skill-iss-location.openvoiceos:where_iss.intent       0.98      1.00      0.99       110
                                   ovos-skill-laugh.openvoiceos:Laugh.intent       0.85      0.86      0.85        51
                             ovos-skill-laugh.openvoiceos:RandomLaugh.intent       0.87      0.76      0.81        34
                                 ovos-skill-laugh.openvoiceos:haunted.intent       0.83      0.83      0.83         6
                ovos-skill-moviemaster.openvoiceos:genre.movie.search.intent       0.68      0.65      0.67        23
                   ovos-skill-moviemaster.openvoiceos:genre.tv.search.intent       1.00      1.00      1.00        55
                        ovos-skill-moviemaster.openvoiceos:movie.cast.intent       1.00      0.92      0.96        13
                 ovos-skill-moviemaster.openvoiceos:movie.description.intent       1.00      0.93      0.96        59
                ovos-skill-moviemaster.openvoiceos:movie.genre.search.intent       0.53      0.56      0.55        16
                      ovos-skill-moviemaster.openvoiceos:movie.genres.intent       1.00      1.00      1.00        78
                 ovos-skill-moviemaster.openvoiceos:movie.information.intent       0.96      1.00      0.98       160
                     ovos-skill-moviemaster.openvoiceos:movie.popular.intent       1.00      0.98      0.99        42
                  ovos-skill-moviemaster.openvoiceos:movie.production.intent       0.89      1.00      0.94         8
             ovos-skill-moviemaster.openvoiceos:movie.recommendations.intent       1.00      1.00      1.00       159
                     ovos-skill-moviemaster.openvoiceos:movie.runtime.intent       1.00      1.00      1.00        40
                         ovos-skill-moviemaster.openvoiceos:movie.top.intent       0.99      0.99      0.99       160
                        ovos-skill-moviemaster.openvoiceos:movie.year.intent       1.00      0.97      0.98        32
                                       ovos-skill-naptime.openvoiceos:WakeUp       0.67      0.33      0.44         6
                               ovos-skill-naptime.openvoiceos:naptime.intent       0.93      0.91      0.92        56
                              ovos-skill-news.openvoiceos:global_news.intent       0.96      0.98      0.97        93
                                     ovos-skill-news.openvoiceos:news.intent       0.99      0.98      0.98       160
                        ovos-skill-parrot.openvoiceos:did.you.hear.me.intent       0.65      0.74      0.69        27
                             ovos-skill-parrot.openvoiceos:repeat.stt.intent       0.82      0.82      0.82        28
                             ovos-skill-parrot.openvoiceos:repeat.tts.intent       0.84      0.94      0.89        69
                                  ovos-skill-parrot.openvoiceos:speak.intent       0.93      0.93      0.93        14
                           ovos-skill-parrot.openvoiceos:start_parrot.intent       0.76      0.85      0.80        33
                            ovos-skill-parrot.openvoiceos:stop_parrot.intent       0.98      0.91      0.94        53
                           ovos-skill-personal.openvoiceos:WhatAreYou.intent       0.81      0.76      0.78        45
                      ovos-skill-personal.openvoiceos:WhenWereYouBorn.intent       0.83      0.95      0.89        42
                     ovos-skill-personal.openvoiceos:WhereWereYouBorn.intent       0.83      0.83      0.83        41
                            ovos-skill-personal.openvoiceos:WhoAreYou.intent       0.65      0.58      0.61        26
                           ovos-skill-personal.openvoiceos:WhoMadeYou.intent       0.95      0.92      0.94        64
                        ovos-skill-randomness.openvoiceos:flip-a-coin.intent       0.83      0.91      0.87        11
                     ovos-skill-randomness.openvoiceos:fortune-teller.intent       0.86      0.75      0.80         8
                      ovos-skill-randomness.openvoiceos:make-a-choice.intent       1.00      0.25      0.40         4
                      ovos-skill-randomness.openvoiceos:pick-a-number.intent       1.00      1.00      1.00         4
                 ovos-skill-randomness.openvoiceos:roll-multiple-dice.intent       1.00      1.00      1.00         6
                    ovos-skill-randomness.openvoiceos:roll-single-die.intent       1.00      1.00      1.00         5
                    ovos-skill-screenshot.openvoiceos:take.screenshot.intent       1.00      0.93      0.96        14
                            ovos-skill-speedtest.openvoiceos:SpeedtestIntent       1.00      0.50      0.67         2
                                 ovos-skill-volume.openvoiceos:change_volume       1.00      0.33      0.50         6
                               ovos-skill-volume.openvoiceos:increase_volume       0.62      0.76      0.68        17
                                   ovos-skill-volume.openvoiceos:less_volume       0.57      0.27      0.36        15
                         ovos-skill-volume.openvoiceos:volume.default.intent       0.89      0.97      0.93        34
                            ovos-skill-volume.openvoiceos:volume.high.intent       0.66      0.95      0.78        20
                             ovos-skill-volume.openvoiceos:volume.low.intent       0.95      0.95      0.95        20
                             ovos-skill-volume.openvoiceos:volume.max.intent       1.00      0.87      0.93        23
                            ovos-skill-volume.openvoiceos:volume.mute.intent       0.73      0.89      0.80        18
                     ovos-skill-volume.openvoiceos:volume.mute.toggle.intent       1.00      0.67      0.80         9
                          ovos-skill-volume.openvoiceos:volume.unmute.intent       0.95      0.95      0.95        22
                       ovos-skill-wallpapers.openvoiceos:MakeWallpaperIntent       0.67      0.40      0.50         5
                      ovos-skill-wallpapers.openvoiceos:picture.about.intent       0.99      1.00      0.99       160
                     ovos-skill-wallpapers.openvoiceos:picture.random.intent       0.98      0.98      0.98       160
                    ovos-skill-wallpapers.openvoiceos:wallpaper.about.intent       1.00      0.99      1.00       160
                   ovos-skill-wallpapers.openvoiceos:wallpaper.random.intent       0.99      1.00      1.00       160
                              ovos-skill-weather.openvoiceos:N_days_forecast       0.78      0.70      0.74        10
                       ovos-skill-weather.openvoiceos:N_days_forecast.intent       0.97      0.97      0.97       160
                                    ovos-skill-weather.openvoiceos:condition       0.25      0.14      0.18         7
                          ovos-skill-weather.openvoiceos:current_temperature       1.00      0.67      0.80         6
                   ovos-skill-weather.openvoiceos:current_temperature.intent       0.94      0.95      0.94       124
                              ovos-skill-weather.openvoiceos:current_weather       1.00      0.75      0.86         4
                       ovos-skill-weather.openvoiceos:current_weather.intent       0.92      0.97      0.95       160
                               ovos-skill-weather.openvoiceos:daily_forecast       0.80      1.00      0.89         4
                        ovos-skill-weather.openvoiceos:daily_forecast.intent       0.89      0.89      0.89       160
                 ovos-skill-weather.openvoiceos:daily_forecast.intent.intent       0.91      0.83      0.87       159
                 ovos-skill-weather.openvoiceos:do.i.need.an.umbrella.intent       0.00      0.00      0.00         2
                                     ovos-skill-weather.openvoiceos:forecast       0.43      0.43      0.43         7
                             ovos-skill-weather.openvoiceos:high_temperature       1.00      0.33      0.50         3
                      ovos-skill-weather.openvoiceos:high_temperature.intent       0.99      0.99      0.99       160
                              ovos-skill-weather.openvoiceos:hourly_forecast       1.00      0.50      0.67         4
                       ovos-skill-weather.openvoiceos:hourly_forecast.intent       0.97      0.99      0.98       160
                           ovos-skill-weather.openvoiceos:hourly_temperature       0.54      0.78      0.64         9
                    ovos-skill-weather.openvoiceos:hourly_temperature.intent       0.99      0.99      0.99       160
                                     ovos-skill-weather.openvoiceos:humidity       1.00      1.00      1.00         3
                              ovos-skill-weather.openvoiceos:humidity.intent       0.96      0.94      0.95        48
                                     ovos-skill-weather.openvoiceos:is_clear       0.86      0.86      0.86         7
                              ovos-skill-weather.openvoiceos:is_clear.intent       0.96      0.95      0.95        95
                                       ovos-skill-weather.openvoiceos:is_fog       0.75      0.75      0.75         4
                                ovos-skill-weather.openvoiceos:is_fog.intent       0.97      0.99      0.98        87
                                      ovos-skill-weather.openvoiceos:is_snow       1.00      1.00      1.00         3
                               ovos-skill-weather.openvoiceos:is_snow.intent       0.98      0.97      0.98        67
                                    ovos-skill-weather.openvoiceos:is_stormy       1.00      0.75      0.86         4
                             ovos-skill-weather.openvoiceos:is_stormy.intent       0.93      0.96      0.94        95
                                      ovos-skill-weather.openvoiceos:is_wind       1.00      1.00      1.00         5
                               ovos-skill-weather.openvoiceos:is_wind.intent       1.00      1.00      1.00       152
                              ovos-skill-weather.openvoiceos:low_temperature       0.50      0.67      0.57         3
                       ovos-skill-weather.openvoiceos:low_temperature.intent       0.98      1.00      0.99       160
                                    ovos-skill-weather.openvoiceos:next_rain       1.00      1.00      1.00         4
                             ovos-skill-weather.openvoiceos:next_rain.intent       1.00      0.88      0.94        34
                                      ovos-skill-weather.openvoiceos:sunrise       1.00      0.88      0.93         8
                               ovos-skill-weather.openvoiceos:sunrise.intent       0.95      0.95      0.95        65
                                       ovos-skill-weather.openvoiceos:sunset       0.60      0.75      0.67         8
                                ovos-skill-weather.openvoiceos:sunset.intent       0.92      0.95      0.94        62
                             ovos-skill-weather.openvoiceos:weekend_forecast       1.00      1.00      1.00         5
                      ovos-skill-weather.openvoiceos:weekend_forecast.intent       0.96      1.00      0.98       160
                               ovos-skill-wikihow.openvoiceos:wikihow.intent       0.98      1.00      0.99        61
                               ovos-skill-wikipedia.openvoiceos:common_query       0.50      0.67      0.57         6
                                ovos-skill-wikipedia.openvoiceos:wiki.intent       0.99      0.97      0.98       116
                        ovos-skill-wikipedia.openvoiceos:wikiroulette.intent       0.97      1.00      0.98       128
                          ovos-skill-wolfie.openvoiceos:search_wolfie.intent       1.00      1.00      1.00        68
                               ovos-skill-wordnet.openvoiceos:antonym.intent       1.00      0.82      0.90        17
                            ovos-skill-wordnet.openvoiceos:definition.intent       0.86      0.85      0.86       160
                               ovos-skill-wordnet.openvoiceos:holonym.intent       1.00      0.88      0.93        16
                              ovos-skill-wordnet.openvoiceos:hypernym.intent       0.94      0.94      0.94        33
                               ovos-skill-wordnet.openvoiceos:hyponym.intent       1.00      0.97      0.98        65
                                 ovos-skill-wordnet.openvoiceos:lemma.intent       0.87      1.00      0.93        13
                        ovos-skill-wordnet.openvoiceos:search_wordnet.intent       0.88      0.95      0.91        22
                               ovos-skill-wordnet.openvoiceos:synonym.intent       0.88      1.00      0.93         7
                                                                   stop:stop       0.91      0.89      0.90        83

                                                                    accuracy                           0.95     10516
                                                                   macro avg       0.90      0.85      0.86     10516
                                                                weighted avg       0.95      0.95      0.95     10516

```

## Per-language evaluation

| Language | Samples | Accuracy | Weighted F1 | Throughput (sps) |
|---|---|---|---|---|
| ca | 2955 | 0.9780 | 0.9776 | 24140 |
| da | 527 | 0.9488 | 0.9465 | 22408 |
| de | 864 | 0.9456 | 0.9431 | 25878 |
| en | 1235 | 0.9263 | 0.9226 | 23515 |
| es | 1163 | 0.9424 | 0.9409 | 24268 |
| eu | 230 | 0.9087 | 0.9017 | 23348 |
| fr | 202 | 0.8762 | 0.8675 | 22476 |
| gl | 913 | 0.9759 | 0.9760 | 25794 |
| it | 682 | 0.9619 | 0.9575 | 22908 |
| nl | 355 | 0.8282 | 0.8233 | 21756 |
| pt | 1390 | 0.9439 | 0.9443 | 25514 |

