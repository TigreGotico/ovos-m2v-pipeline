# Model Evaluation – potion-multilingual-128M

**Base model:** `minishlab/potion-multilingual-128M`  
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
| Disk size | 518.49 MB |
| Parameters | 129.09 M |

## Benchmark

| Metric | Value |
|---|---|
| Training time | 108.2s |
| Inference time (10516 samples) | 0.398s |
| Throughput | 26393 sps |

## Overall evaluation

| Metric | Value |
|---|---|
| Accuracy | 0.9603 |
| Weighted F1 | 0.9595 |

### Classification report

```
                                                                              precision    recall  f1-score   support

                                                   common_query:common_query       0.82      0.79      0.81       160
                                                    common_query:wiki.intent       0.33      0.17      0.22         6
                                                                    ocp:play       0.98      0.97      0.97       160
                               ovos-skill-alerts.openvoiceos:AddListSubitems       0.88      0.93      0.90        15
                                  ovos-skill-alerts.openvoiceos:CalendarList       1.00      1.00      1.00         5
                                   ovos-skill-alerts.openvoiceos:CancelAlert       1.00      0.73      0.84        11
                              ovos-skill-alerts.openvoiceos:ChangeProperties       1.00      0.71      0.83         7
                                   ovos-skill-alerts.openvoiceos:CreateAlarm       0.89      0.89      0.89        19
                                   ovos-skill-alerts.openvoiceos:CreateEvent       0.86      1.00      0.92         6
                                    ovos-skill-alerts.openvoiceos:CreateList       1.00      1.00      1.00         8
                                ovos-skill-alerts.openvoiceos:CreateOcpAlarm       0.96      0.94      0.95        47
                                ovos-skill-alerts.openvoiceos:CreateReminder       0.92      0.92      0.92        12
                                   ovos-skill-alerts.openvoiceos:CreateTimer       1.00      1.00      1.00         7
                                       ovos-skill-alerts.openvoiceos:DAVSync       1.00      0.60      0.75         5
                                    ovos-skill-alerts.openvoiceos:DeleteList       1.00      1.00      1.00         8
                             ovos-skill-alerts.openvoiceos:DeleteListEntries       0.91      1.00      0.95        10
                             ovos-skill-alerts.openvoiceos:DeleteTodoEntries       0.82      0.90      0.86        10
                                    ovos-skill-alerts.openvoiceos:ListAlerts       0.90      0.95      0.93        20
                              ovos-skill-alerts.openvoiceos:QueryListEntries       1.00      0.60      0.75         5
                                ovos-skill-alerts.openvoiceos:QueryListNames       1.00      1.00      1.00         5
                              ovos-skill-alerts.openvoiceos:QueryTodoEntries       1.00      1.00      1.00         5
                               ovos-skill-alerts.openvoiceos:RescheduleAlert       0.78      0.78      0.78         9
                                   ovos-skill-alerts.openvoiceos:TimerStatus       0.89      1.00      0.94        24
                          ovos-skill-alerts.openvoiceos:missed_alerts.intent       0.99      0.99      0.99       160
                    ovos-skill-application-launcher.openvoiceos:close.intent       0.89      1.00      0.94         8
                   ovos-skill-application-launcher.openvoiceos:launch.intent       1.00      0.90      0.95        10
               ovos-skill-audio-recording.openvoiceos:start_recording.intent       0.98      1.00      0.99       160
                   ovos-skill-boot-finished.openvoiceos:are_you_ready.intent       0.96      0.99      0.98       123
      ovos-skill-boot-finished.openvoiceos:disable_ready_notification.intent       0.97      0.99      0.98       160
       ovos-skill-boot-finished.openvoiceos:enable_ready_notification.intent       0.99      0.98      0.99       160
                            ovos-skill-camera.openvoiceos:have_camera.intent       1.00      1.00      1.00        12
                           ovos-skill-camera.openvoiceos:take_picture.intent       0.81      1.00      0.90        13
ovos-skill-color-picker.krisgesling.openvoiceos:request-color-by-name.intent       1.00      0.83      0.91         6
             ovos-skill-color-picker.krisgesling:request-color-by-hex.intent       1.00      1.00      1.00         2
            ovos-skill-color-picker.krisgesling:request-color-by-name.intent       1.00      1.00      1.00         6
                      ovos-skill-confucius-quotes.openvoiceos:ConfuciusBirth       1.00      0.86      0.92         7
                      ovos-skill-confucius-quotes.openvoiceos:ConfuciusDeath       1.00      0.86      0.92         7
                      ovos-skill-confucius-quotes.openvoiceos:ConfuciusQuote       0.89      1.00      0.94         8
                          ovos-skill-confucius-quotes.openvoiceos:who.intent       1.00      1.00      1.00         5
                              ovos-skill-count.openvoiceos:count_to_N.intent       1.00      1.00      1.00       160
                        ovos-skill-date-time.openvoiceos:current_date.intent       0.95      1.00      0.98       160
                 ovos-skill-date-time.openvoiceos:date.future.weekend.intent       1.00      0.98      0.99       160
                   ovos-skill-date-time.openvoiceos:date.last.weekend.intent       0.99      0.99      0.99       160
                        ovos-skill-date-time.openvoiceos:handle_day_for_date       0.95      0.95      0.95        19
                      ovos-skill-date-time.openvoiceos:next.leap.year.intent       0.99      1.00      0.99        77
                          ovos-skill-date-time.openvoiceos:time.until.intent       1.00      1.00      1.00        84
                    ovos-skill-date-time.openvoiceos:weekday.for.date.intent       0.99      0.99      0.99       160
                      ovos-skill-date-time.openvoiceos:what.day.is.it.intent       0.83      0.92      0.88        53
                    ovos-skill-date-time.openvoiceos:what.month.is.it.intent       1.00      0.99      1.00       160
                     ovos-skill-date-time.openvoiceos:what.time.is.it.intent       0.96      0.97      0.97       160
                ovos-skill-date-time.openvoiceos:what.time.will.it.be.intent       1.00      0.99      0.99       160
                  ovos-skill-date-time.openvoiceos:what.weekday.is.it.intent       0.98      0.89      0.93        71
                     ovos-skill-date-time.openvoiceos:what.year.is.it.intent       1.00      1.00      1.00        32
             ovos-skill-days-in-history.openvoiceos:births_in_history.intent       1.00      0.75      0.86        12
             ovos-skill-days-in-history.openvoiceos:deaths_in_history.intent       1.00      1.00      1.00         8
              ovos-skill-days-in-history.openvoiceos:today_in_history.intent       0.99      0.99      0.99       160
                              ovos-skill-ddg.openvoiceos:age_at_death.intent       1.00      1.00      1.00         9
                                 ovos-skill-ddg.openvoiceos:birthdate.intent       0.29      0.25      0.27         8
                                      ovos-skill-ddg.openvoiceos:born.intent       0.42      0.56      0.48         9
                                  ovos-skill-ddg.openvoiceos:children.intent       1.00      1.00      1.00         4
                                      ovos-skill-ddg.openvoiceos:died.intent       1.00      0.89      0.94         9
                                 ovos-skill-ddg.openvoiceos:education.intent       1.00      0.60      0.75         5
                                 ovos-skill-ddg.openvoiceos:known_for.intent       0.67      0.50      0.57         4
                          ovos-skill-ddg.openvoiceos:official_website.intent       1.00      0.50      0.67         2
                             ovos-skill-ddg.openvoiceos:resting_place.intent       1.00      1.00      1.00         4
                               ovos-skill-ddg.openvoiceos:search_duck.intent       0.98      1.00      0.99        40
                                       ovos-skill-ddg.openvoiceos:who.intent       1.00      1.00      1.00         2
                   ovos-skill-diagnostics.openvoiceos:query_cpu_usage.intent       1.00      1.00      1.00        25
                 ovos-skill-diagnostics.openvoiceos:query_extra_langs.intent       0.98      1.00      0.99        43
                         ovos-skill-diagnostics.openvoiceos:query_gpu.intent       1.00      1.00      1.00        14
              ovos-skill-diagnostics.openvoiceos:query_kernel_version.intent       1.00      1.00      1.00        14
                       ovos-skill-diagnostics.openvoiceos:query_langs.intent       1.00      0.99      0.99       160
                ovos-skill-diagnostics.openvoiceos:query_memory_usage.intent       1.00      1.00      1.00        26
               ovos-skill-diagnostics.openvoiceos:query_ovos_location.intent       0.98      1.00      0.99        48
                ovos-skill-diagnostics.openvoiceos:query_primary_lang.intent       0.98      0.99      0.98        93
                   ovos-skill-diagnostics.openvoiceos:query_user_lang.intent       0.95      1.00      0.97        37
               ovos-skill-diagnostics.openvoiceos:query_user_location.intent       1.00      0.96      0.98        26
                     ovos-skill-dictation.openvoiceos:start_dictation.intent       0.99      0.97      0.98       160
                      ovos-skill-dictation.openvoiceos:stop_dictation.intent       0.99      0.99      0.99       117
                   ovos-skill-fuster-quotes.openvoiceos:fuster_quotes.intent       1.00      1.00      1.00        12
                             ovos-skill-fuster-quotes.openvoiceos:who.intent       1.00      1.00      1.00         5
                         ovos-skill-hello-world.openvoiceos:Greetings.intent       0.89      0.84      0.86        97
                           ovos-skill-hello-world.openvoiceos:ThankYouIntent       1.00      1.00      1.00         7
                          ovos-skill-icanhazdadjokes.openvoiceos:joke.intent       0.97      0.97      0.97       160
                   ovos-skill-icanhazdadjokes.openvoiceos:search_joke.intent       1.00      0.97      0.98        29
                                          ovos-skill-ip.openvoiceos:IPIntent       1.00      1.00      1.00         9
                                  ovos-skill-ip.openvoiceos:what.ssid.intent       1.00      1.00      1.00        96
                         ovos-skill-iss-location.openvoiceos:NumberISSIntent       1.00      1.00      1.00         6
                            ovos-skill-iss-location.openvoiceos:WhoISSIntent       1.00      1.00      1.00        10
                         ovos-skill-iss-location.openvoiceos:when_iss.intent       1.00      1.00      1.00       160
                        ovos-skill-iss-location.openvoiceos:where_iss.intent       1.00      1.00      1.00       110
                                   ovos-skill-laugh.openvoiceos:Laugh.intent       0.89      0.94      0.91        51
                             ovos-skill-laugh.openvoiceos:RandomLaugh.intent       0.97      0.82      0.89        34
                                 ovos-skill-laugh.openvoiceos:haunted.intent       1.00      0.83      0.91         6
                ovos-skill-moviemaster.openvoiceos:genre.movie.search.intent       0.68      0.83      0.75        23
                   ovos-skill-moviemaster.openvoiceos:genre.tv.search.intent       0.98      1.00      0.99        55
                        ovos-skill-moviemaster.openvoiceos:movie.cast.intent       1.00      1.00      1.00        13
                 ovos-skill-moviemaster.openvoiceos:movie.description.intent       1.00      1.00      1.00        59
                ovos-skill-moviemaster.openvoiceos:movie.genre.search.intent       0.64      0.44      0.52        16
                      ovos-skill-moviemaster.openvoiceos:movie.genres.intent       1.00      1.00      1.00        78
                 ovos-skill-moviemaster.openvoiceos:movie.information.intent       0.99      1.00      0.99       160
                     ovos-skill-moviemaster.openvoiceos:movie.popular.intent       0.98      1.00      0.99        42
                  ovos-skill-moviemaster.openvoiceos:movie.production.intent       1.00      1.00      1.00         8
             ovos-skill-moviemaster.openvoiceos:movie.recommendations.intent       1.00      1.00      1.00       159
                     ovos-skill-moviemaster.openvoiceos:movie.runtime.intent       1.00      1.00      1.00        40
                         ovos-skill-moviemaster.openvoiceos:movie.top.intent       0.99      1.00      0.99       160
                        ovos-skill-moviemaster.openvoiceos:movie.year.intent       1.00      0.97      0.98        32
                                       ovos-skill-naptime.openvoiceos:WakeUp       0.33      0.17      0.22         6
                               ovos-skill-naptime.openvoiceos:naptime.intent       0.92      0.98      0.95        56
                              ovos-skill-news.openvoiceos:global_news.intent       0.97      0.99      0.98        93
                                     ovos-skill-news.openvoiceos:news.intent       0.99      0.99      0.99       160
                        ovos-skill-parrot.openvoiceos:did.you.hear.me.intent       0.75      0.89      0.81        27
                             ovos-skill-parrot.openvoiceos:repeat.stt.intent       0.79      0.82      0.81        28
                             ovos-skill-parrot.openvoiceos:repeat.tts.intent       0.88      0.97      0.92        69
                                  ovos-skill-parrot.openvoiceos:speak.intent       0.87      0.93      0.90        14
                           ovos-skill-parrot.openvoiceos:start_parrot.intent       0.91      0.91      0.91        33
                            ovos-skill-parrot.openvoiceos:stop_parrot.intent       0.96      0.91      0.93        53
                           ovos-skill-personal.openvoiceos:WhatAreYou.intent       0.85      0.78      0.81        45
                      ovos-skill-personal.openvoiceos:WhenWereYouBorn.intent       0.88      1.00      0.93        42
                     ovos-skill-personal.openvoiceos:WhereWereYouBorn.intent       0.92      0.80      0.86        41
                            ovos-skill-personal.openvoiceos:WhoAreYou.intent       0.60      0.58      0.59        26
                           ovos-skill-personal.openvoiceos:WhoMadeYou.intent       0.95      0.95      0.95        64
                        ovos-skill-randomness.openvoiceos:flip-a-coin.intent       1.00      0.82      0.90        11
                     ovos-skill-randomness.openvoiceos:fortune-teller.intent       1.00      0.88      0.93         8
                      ovos-skill-randomness.openvoiceos:make-a-choice.intent       1.00      0.75      0.86         4
                      ovos-skill-randomness.openvoiceos:pick-a-number.intent       1.00      1.00      1.00         4
                 ovos-skill-randomness.openvoiceos:roll-multiple-dice.intent       1.00      1.00      1.00         6
                    ovos-skill-randomness.openvoiceos:roll-single-die.intent       1.00      1.00      1.00         5
                    ovos-skill-screenshot.openvoiceos:take.screenshot.intent       0.93      0.93      0.93        14
                            ovos-skill-speedtest.openvoiceos:SpeedtestIntent       1.00      1.00      1.00         2
                                 ovos-skill-volume.openvoiceos:change_volume       1.00      0.50      0.67         6
                               ovos-skill-volume.openvoiceos:increase_volume       0.65      0.76      0.70        17
                                   ovos-skill-volume.openvoiceos:less_volume       0.90      0.60      0.72        15
                         ovos-skill-volume.openvoiceos:volume.default.intent       1.00      1.00      1.00        34
                            ovos-skill-volume.openvoiceos:volume.high.intent       0.70      0.95      0.81        20
                             ovos-skill-volume.openvoiceos:volume.low.intent       0.90      0.95      0.93        20
                             ovos-skill-volume.openvoiceos:volume.max.intent       0.95      0.91      0.93        23
                            ovos-skill-volume.openvoiceos:volume.mute.intent       0.89      0.89      0.89        18
                     ovos-skill-volume.openvoiceos:volume.mute.toggle.intent       1.00      0.44      0.62         9
                          ovos-skill-volume.openvoiceos:volume.unmute.intent       1.00      0.91      0.95        22
                       ovos-skill-wallpapers.openvoiceos:MakeWallpaperIntent       1.00      1.00      1.00         5
                      ovos-skill-wallpapers.openvoiceos:picture.about.intent       1.00      1.00      1.00       160
                     ovos-skill-wallpapers.openvoiceos:picture.random.intent       1.00      0.98      0.99       160
                    ovos-skill-wallpapers.openvoiceos:wallpaper.about.intent       1.00      0.99      1.00       160
                   ovos-skill-wallpapers.openvoiceos:wallpaper.random.intent       1.00      1.00      1.00       160
                              ovos-skill-weather.openvoiceos:N_days_forecast       0.64      0.90      0.75        10
                       ovos-skill-weather.openvoiceos:N_days_forecast.intent       0.98      0.97      0.97       160
                                    ovos-skill-weather.openvoiceos:condition       0.33      0.14      0.20         7
                          ovos-skill-weather.openvoiceos:current_temperature       1.00      0.67      0.80         6
                   ovos-skill-weather.openvoiceos:current_temperature.intent       0.92      0.98      0.95       124
                              ovos-skill-weather.openvoiceos:current_weather       0.75      0.75      0.75         4
                       ovos-skill-weather.openvoiceos:current_weather.intent       0.92      0.97      0.95       160
                               ovos-skill-weather.openvoiceos:daily_forecast       1.00      0.75      0.86         4
                        ovos-skill-weather.openvoiceos:daily_forecast.intent       0.89      0.92      0.90       160
                 ovos-skill-weather.openvoiceos:daily_forecast.intent.intent       0.89      0.86      0.87       159
                 ovos-skill-weather.openvoiceos:do.i.need.an.umbrella.intent       0.00      0.00      0.00         2
                                     ovos-skill-weather.openvoiceos:forecast       1.00      0.57      0.73         7
                             ovos-skill-weather.openvoiceos:high_temperature       1.00      0.67      0.80         3
                      ovos-skill-weather.openvoiceos:high_temperature.intent       0.99      0.98      0.99       160
                              ovos-skill-weather.openvoiceos:hourly_forecast       1.00      0.75      0.86         4
                       ovos-skill-weather.openvoiceos:hourly_forecast.intent       0.98      1.00      0.99       160
                           ovos-skill-weather.openvoiceos:hourly_temperature       0.75      0.67      0.71         9
                    ovos-skill-weather.openvoiceos:hourly_temperature.intent       0.97      0.99      0.98       160
                                     ovos-skill-weather.openvoiceos:humidity       1.00      1.00      1.00         3
                              ovos-skill-weather.openvoiceos:humidity.intent       1.00      0.94      0.97        48
                                     ovos-skill-weather.openvoiceos:is_clear       1.00      1.00      1.00         7
                              ovos-skill-weather.openvoiceos:is_clear.intent       0.95      0.95      0.95        95
                                       ovos-skill-weather.openvoiceos:is_fog       0.57      1.00      0.73         4
                                ovos-skill-weather.openvoiceos:is_fog.intent       1.00      0.94      0.97        87
                                      ovos-skill-weather.openvoiceos:is_snow       1.00      1.00      1.00         3
                               ovos-skill-weather.openvoiceos:is_snow.intent       0.97      1.00      0.99        67
                                    ovos-skill-weather.openvoiceos:is_stormy       1.00      0.75      0.86         4
                             ovos-skill-weather.openvoiceos:is_stormy.intent       0.99      0.94      0.96        95
                                      ovos-skill-weather.openvoiceos:is_wind       1.00      1.00      1.00         5
                               ovos-skill-weather.openvoiceos:is_wind.intent       1.00      1.00      1.00       152
                              ovos-skill-weather.openvoiceos:low_temperature       0.75      1.00      0.86         3
                       ovos-skill-weather.openvoiceos:low_temperature.intent       0.98      0.99      0.99       160
                                    ovos-skill-weather.openvoiceos:next_rain       1.00      0.75      0.86         4
                             ovos-skill-weather.openvoiceos:next_rain.intent       0.94      0.91      0.93        34
                                      ovos-skill-weather.openvoiceos:sunrise       1.00      0.75      0.86         8
                               ovos-skill-weather.openvoiceos:sunrise.intent       0.93      0.95      0.94        65
                                       ovos-skill-weather.openvoiceos:sunset       1.00      0.75      0.86         8
                                ovos-skill-weather.openvoiceos:sunset.intent       0.94      0.95      0.94        62
                             ovos-skill-weather.openvoiceos:weekend_forecast       1.00      1.00      1.00         5
                      ovos-skill-weather.openvoiceos:weekend_forecast.intent       0.96      1.00      0.98       160
                               ovos-skill-wikihow.openvoiceos:wikihow.intent       1.00      1.00      1.00        61
                               ovos-skill-wikipedia.openvoiceos:common_query       0.56      0.83      0.67         6
                                ovos-skill-wikipedia.openvoiceos:wiki.intent       0.99      0.97      0.98       116
                        ovos-skill-wikipedia.openvoiceos:wikiroulette.intent       0.98      1.00      0.99       128
                          ovos-skill-wolfie.openvoiceos:search_wolfie.intent       1.00      1.00      1.00        68
                               ovos-skill-wordnet.openvoiceos:antonym.intent       1.00      0.88      0.94        17
                            ovos-skill-wordnet.openvoiceos:definition.intent       0.86      0.85      0.86       160
                               ovos-skill-wordnet.openvoiceos:holonym.intent       1.00      0.88      0.93        16
                              ovos-skill-wordnet.openvoiceos:hypernym.intent       0.94      0.94      0.94        33
                               ovos-skill-wordnet.openvoiceos:hyponym.intent       1.00      1.00      1.00        65
                                 ovos-skill-wordnet.openvoiceos:lemma.intent       0.93      1.00      0.96        13
                        ovos-skill-wordnet.openvoiceos:search_wordnet.intent       0.95      0.95      0.95        22
                               ovos-skill-wordnet.openvoiceos:synonym.intent       0.88      1.00      0.93         7
                                                                   stop:stop       0.90      0.93      0.91        83

                                                                    accuracy                           0.96     10516
                                                                   macro avg       0.93      0.90      0.91     10516
                                                                weighted avg       0.96      0.96      0.96     10516

```

## Per-language evaluation

| Language | Samples | Accuracy | Weighted F1 | Throughput (sps) |
|---|---|---|---|---|
| ca | 2955 | 0.9834 | 0.9827 | 28891 |
| da | 527 | 0.9677 | 0.9677 | 29656 |
| de | 864 | 0.9641 | 0.9628 | 31430 |
| en | 1235 | 0.9377 | 0.9374 | 29360 |
| es | 1163 | 0.9561 | 0.9549 | 25964 |
| eu | 230 | 0.9174 | 0.9101 | 30378 |
| fr | 202 | 0.8960 | 0.8881 | 27828 |
| gl | 913 | 0.9803 | 0.9807 | 29963 |
| it | 682 | 0.9751 | 0.9697 | 30978 |
| nl | 355 | 0.8423 | 0.8365 | 28519 |
| pt | 1390 | 0.9554 | 0.9558 | 25227 |

