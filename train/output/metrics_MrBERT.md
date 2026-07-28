# Model Evaluation – MrBERT

**Base model:** `/home/miro/PycharmProjects/ovos-m2v-pipeline/train/output/distilled/MrBERT`  
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
| Disk size | 267.44 MB |
| Parameters | 66.01 M |

## Benchmark

| Metric | Value |
|---|---|
| Training time | 72.5s |
| Inference time (10516 samples) | 0.435s |
| Throughput | 24194 sps |

## Overall evaluation

| Metric | Value |
|---|---|
| Accuracy | 0.8241 |
| Weighted F1 | 0.8053 |

### Classification report

```
                                                                              precision    recall  f1-score   support

                                                   common_query:common_query       0.61      0.64      0.63       160
                                                    common_query:wiki.intent       0.20      0.33      0.25         6
                                                                    ocp:play       0.81      0.96      0.88       160
                               ovos-skill-alerts.openvoiceos:AddListSubitems       0.50      0.93      0.65        15
                                  ovos-skill-alerts.openvoiceos:CalendarList       1.00      0.40      0.57         5
                                   ovos-skill-alerts.openvoiceos:CancelAlert       0.00      0.00      0.00        11
                              ovos-skill-alerts.openvoiceos:ChangeProperties       0.43      0.43      0.43         7
                                   ovos-skill-alerts.openvoiceos:CreateAlarm       1.00      0.16      0.27        19
                                   ovos-skill-alerts.openvoiceos:CreateEvent       0.45      0.83      0.59         6
                                    ovos-skill-alerts.openvoiceos:CreateList       1.00      0.25      0.40         8
                                ovos-skill-alerts.openvoiceos:CreateOcpAlarm       0.67      0.96      0.79        47
                                ovos-skill-alerts.openvoiceos:CreateReminder       0.30      0.50      0.38        12
                                   ovos-skill-alerts.openvoiceos:CreateTimer       0.27      0.86      0.41         7
                                       ovos-skill-alerts.openvoiceos:DAVSync       0.00      0.00      0.00         5
                                    ovos-skill-alerts.openvoiceos:DeleteList       0.00      0.00      0.00         8
                             ovos-skill-alerts.openvoiceos:DeleteListEntries       0.54      0.70      0.61        10
                             ovos-skill-alerts.openvoiceos:DeleteTodoEntries       0.00      0.00      0.00        10
                                    ovos-skill-alerts.openvoiceos:ListAlerts       0.62      0.25      0.36        20
                              ovos-skill-alerts.openvoiceos:QueryListEntries       0.00      0.00      0.00         5
                                ovos-skill-alerts.openvoiceos:QueryListNames       0.00      0.00      0.00         5
                              ovos-skill-alerts.openvoiceos:QueryTodoEntries       0.00      0.00      0.00         5
                               ovos-skill-alerts.openvoiceos:RescheduleAlert       0.38      0.67      0.48         9
                                   ovos-skill-alerts.openvoiceos:TimerStatus       0.70      0.58      0.64        24
                          ovos-skill-alerts.openvoiceos:missed_alerts.intent       0.97      0.93      0.95       160
                    ovos-skill-application-launcher.openvoiceos:close.intent       0.60      0.38      0.46         8
                   ovos-skill-application-launcher.openvoiceos:launch.intent       1.00      0.30      0.46        10
               ovos-skill-audio-recording.openvoiceos:start_recording.intent       0.83      0.96      0.89       160
                   ovos-skill-boot-finished.openvoiceos:are_you_ready.intent       0.98      0.80      0.88       123
      ovos-skill-boot-finished.openvoiceos:disable_ready_notification.intent       0.90      0.98      0.94       160
       ovos-skill-boot-finished.openvoiceos:enable_ready_notification.intent       0.98      0.95      0.97       160
                            ovos-skill-camera.openvoiceos:have_camera.intent       0.00      0.00      0.00        12
                           ovos-skill-camera.openvoiceos:take_picture.intent       0.00      0.00      0.00        13
ovos-skill-color-picker.krisgesling.openvoiceos:request-color-by-name.intent       0.00      0.00      0.00         6
             ovos-skill-color-picker.krisgesling:request-color-by-hex.intent       0.50      1.00      0.67         2
            ovos-skill-color-picker.krisgesling:request-color-by-name.intent       1.00      0.33      0.50         6
                      ovos-skill-confucius-quotes.openvoiceos:ConfuciusBirth       0.00      0.00      0.00         7
                      ovos-skill-confucius-quotes.openvoiceos:ConfuciusDeath       0.13      0.57      0.21         7
                      ovos-skill-confucius-quotes.openvoiceos:ConfuciusQuote       0.57      1.00      0.73         8
                          ovos-skill-confucius-quotes.openvoiceos:who.intent       0.00      0.00      0.00         5
                              ovos-skill-count.openvoiceos:count_to_N.intent       0.98      1.00      0.99       160
                        ovos-skill-date-time.openvoiceos:current_date.intent       0.95      0.94      0.95       160
                 ovos-skill-date-time.openvoiceos:date.future.weekend.intent       0.87      0.98      0.92       160
                   ovos-skill-date-time.openvoiceos:date.last.weekend.intent       0.88      0.97      0.93       160
                        ovos-skill-date-time.openvoiceos:handle_day_for_date       0.86      0.63      0.73        19
                      ovos-skill-date-time.openvoiceos:next.leap.year.intent       0.86      0.99      0.92        77
                          ovos-skill-date-time.openvoiceos:time.until.intent       0.97      0.87      0.92        84
                    ovos-skill-date-time.openvoiceos:weekday.for.date.intent       0.67      0.99      0.80       160
                      ovos-skill-date-time.openvoiceos:what.day.is.it.intent       0.54      0.85      0.66        53
                    ovos-skill-date-time.openvoiceos:what.month.is.it.intent       0.93      0.97      0.95       160
                     ovos-skill-date-time.openvoiceos:what.time.is.it.intent       0.97      0.68      0.80       160
                ovos-skill-date-time.openvoiceos:what.time.will.it.be.intent       0.84      0.97      0.90       160
                  ovos-skill-date-time.openvoiceos:what.weekday.is.it.intent       0.94      0.42      0.58        71
                     ovos-skill-date-time.openvoiceos:what.year.is.it.intent       0.42      0.97      0.58        32
             ovos-skill-days-in-history.openvoiceos:births_in_history.intent       0.55      0.50      0.52        12
             ovos-skill-days-in-history.openvoiceos:deaths_in_history.intent       0.00      0.00      0.00         8
              ovos-skill-days-in-history.openvoiceos:today_in_history.intent       0.99      0.99      0.99       160
                              ovos-skill-ddg.openvoiceos:age_at_death.intent       0.70      0.78      0.74         9
                                 ovos-skill-ddg.openvoiceos:birthdate.intent       0.42      1.00      0.59         8
                                      ovos-skill-ddg.openvoiceos:born.intent       0.00      0.00      0.00         9
                                  ovos-skill-ddg.openvoiceos:children.intent       0.00      0.00      0.00         4
                                      ovos-skill-ddg.openvoiceos:died.intent       0.00      0.00      0.00         9
                                 ovos-skill-ddg.openvoiceos:education.intent       0.00      0.00      0.00         5
                                 ovos-skill-ddg.openvoiceos:known_for.intent       0.00      0.00      0.00         4
                          ovos-skill-ddg.openvoiceos:official_website.intent       0.00      0.00      0.00         2
                             ovos-skill-ddg.openvoiceos:resting_place.intent       0.00      0.00      0.00         4
                               ovos-skill-ddg.openvoiceos:search_duck.intent       0.68      0.95      0.79        40
                                       ovos-skill-ddg.openvoiceos:who.intent       0.00      0.00      0.00         2
                   ovos-skill-diagnostics.openvoiceos:query_cpu_usage.intent       0.67      0.88      0.76        25
                 ovos-skill-diagnostics.openvoiceos:query_extra_langs.intent       0.83      0.88      0.85        43
                         ovos-skill-diagnostics.openvoiceos:query_gpu.intent       0.83      0.36      0.50        14
              ovos-skill-diagnostics.openvoiceos:query_kernel_version.intent       0.91      0.71      0.80        14
                       ovos-skill-diagnostics.openvoiceos:query_langs.intent       0.92      0.96      0.94       160
                ovos-skill-diagnostics.openvoiceos:query_memory_usage.intent       0.90      0.35      0.50        26
               ovos-skill-diagnostics.openvoiceos:query_ovos_location.intent       0.94      0.33      0.49        48
                ovos-skill-diagnostics.openvoiceos:query_primary_lang.intent       0.94      0.96      0.95        93
                   ovos-skill-diagnostics.openvoiceos:query_user_lang.intent       0.97      0.81      0.88        37
               ovos-skill-diagnostics.openvoiceos:query_user_location.intent       0.55      0.88      0.68        26
                     ovos-skill-dictation.openvoiceos:start_dictation.intent       0.80      0.86      0.83       160
                      ovos-skill-dictation.openvoiceos:stop_dictation.intent       0.95      0.76      0.84       117
                   ovos-skill-fuster-quotes.openvoiceos:fuster_quotes.intent       0.70      0.58      0.64        12
                             ovos-skill-fuster-quotes.openvoiceos:who.intent       0.83      1.00      0.91         5
                         ovos-skill-hello-world.openvoiceos:Greetings.intent       1.00      0.01      0.02        97
                           ovos-skill-hello-world.openvoiceos:ThankYouIntent       0.00      0.00      0.00         7
                          ovos-skill-icanhazdadjokes.openvoiceos:joke.intent       0.89      0.82      0.86       160
                   ovos-skill-icanhazdadjokes.openvoiceos:search_joke.intent       0.68      0.90      0.78        29
                                          ovos-skill-ip.openvoiceos:IPIntent       0.00      0.00      0.00         9
                                  ovos-skill-ip.openvoiceos:what.ssid.intent       0.78      1.00      0.88        96
                         ovos-skill-iss-location.openvoiceos:NumberISSIntent       0.67      0.33      0.44         6
                            ovos-skill-iss-location.openvoiceos:WhoISSIntent       0.60      0.60      0.60        10
                         ovos-skill-iss-location.openvoiceos:when_iss.intent       0.99      0.99      0.99       160
                        ovos-skill-iss-location.openvoiceos:where_iss.intent       0.93      0.92      0.92       110
                                   ovos-skill-laugh.openvoiceos:Laugh.intent       0.79      0.65      0.71        51
                             ovos-skill-laugh.openvoiceos:RandomLaugh.intent       0.84      0.47      0.60        34
                                 ovos-skill-laugh.openvoiceos:haunted.intent       0.00      0.00      0.00         6
                ovos-skill-moviemaster.openvoiceos:genre.movie.search.intent       1.00      0.09      0.16        23
                   ovos-skill-moviemaster.openvoiceos:genre.tv.search.intent       0.85      1.00      0.92        55
                        ovos-skill-moviemaster.openvoiceos:movie.cast.intent       0.59      0.77      0.67        13
                 ovos-skill-moviemaster.openvoiceos:movie.description.intent       0.81      0.92      0.86        59
                ovos-skill-moviemaster.openvoiceos:movie.genre.search.intent       0.46      0.81      0.59        16
                      ovos-skill-moviemaster.openvoiceos:movie.genres.intent       0.87      1.00      0.93        78
                 ovos-skill-moviemaster.openvoiceos:movie.information.intent       0.95      1.00      0.98       160
                     ovos-skill-moviemaster.openvoiceos:movie.popular.intent       0.89      0.79      0.84        42
                  ovos-skill-moviemaster.openvoiceos:movie.production.intent       0.00      0.00      0.00         8
             ovos-skill-moviemaster.openvoiceos:movie.recommendations.intent       0.95      0.99      0.97       159
                     ovos-skill-moviemaster.openvoiceos:movie.runtime.intent       0.84      0.93      0.88        40
                         ovos-skill-moviemaster.openvoiceos:movie.top.intent       0.90      0.99      0.94       160
                        ovos-skill-moviemaster.openvoiceos:movie.year.intent       0.83      0.94      0.88        32
                                       ovos-skill-naptime.openvoiceos:WakeUp       0.00      0.00      0.00         6
                               ovos-skill-naptime.openvoiceos:naptime.intent       1.00      0.02      0.04        56
                              ovos-skill-news.openvoiceos:global_news.intent       0.71      0.96      0.82        93
                                     ovos-skill-news.openvoiceos:news.intent       0.98      0.83      0.90       160
                        ovos-skill-parrot.openvoiceos:did.you.hear.me.intent       0.50      0.22      0.31        27
                             ovos-skill-parrot.openvoiceos:repeat.stt.intent       0.22      0.68      0.33        28
                             ovos-skill-parrot.openvoiceos:repeat.tts.intent       0.55      0.68      0.61        69
                                  ovos-skill-parrot.openvoiceos:speak.intent       0.75      0.86      0.80        14
                           ovos-skill-parrot.openvoiceos:start_parrot.intent       0.74      0.61      0.67        33
                            ovos-skill-parrot.openvoiceos:stop_parrot.intent       0.73      0.85      0.78        53
                           ovos-skill-personal.openvoiceos:WhatAreYou.intent       0.82      0.31      0.45        45
                      ovos-skill-personal.openvoiceos:WhenWereYouBorn.intent       0.70      0.83      0.76        42
                     ovos-skill-personal.openvoiceos:WhereWereYouBorn.intent       0.70      0.39      0.50        41
                            ovos-skill-personal.openvoiceos:WhoAreYou.intent       1.00      0.08      0.14        26
                           ovos-skill-personal.openvoiceos:WhoMadeYou.intent       0.83      0.67      0.74        64
                        ovos-skill-randomness.openvoiceos:flip-a-coin.intent       0.00      0.00      0.00        11
                     ovos-skill-randomness.openvoiceos:fortune-teller.intent       0.00      0.00      0.00         8
                      ovos-skill-randomness.openvoiceos:make-a-choice.intent       0.00      0.00      0.00         4
                      ovos-skill-randomness.openvoiceos:pick-a-number.intent       0.27      1.00      0.42         4
                 ovos-skill-randomness.openvoiceos:roll-multiple-dice.intent       0.75      1.00      0.86         6
                    ovos-skill-randomness.openvoiceos:roll-single-die.intent       0.00      0.00      0.00         5
                    ovos-skill-screenshot.openvoiceos:take.screenshot.intent       0.00      0.00      0.00        14
                            ovos-skill-speedtest.openvoiceos:SpeedtestIntent       0.00      0.00      0.00         2
                                 ovos-skill-volume.openvoiceos:change_volume       0.33      0.33      0.33         6
                               ovos-skill-volume.openvoiceos:increase_volume       0.00      0.00      0.00        17
                                   ovos-skill-volume.openvoiceos:less_volume       0.00      0.00      0.00        15
                         ovos-skill-volume.openvoiceos:volume.default.intent       0.84      0.79      0.82        34
                            ovos-skill-volume.openvoiceos:volume.high.intent       0.52      0.80      0.63        20
                             ovos-skill-volume.openvoiceos:volume.low.intent       1.00      0.80      0.89        20
                             ovos-skill-volume.openvoiceos:volume.max.intent       0.74      0.87      0.80        23
                            ovos-skill-volume.openvoiceos:volume.mute.intent       0.00      0.00      0.00        18
                     ovos-skill-volume.openvoiceos:volume.mute.toggle.intent       0.00      0.00      0.00         9
                          ovos-skill-volume.openvoiceos:volume.unmute.intent       0.52      0.50      0.51        22
                       ovos-skill-wallpapers.openvoiceos:MakeWallpaperIntent       0.00      0.00      0.00         5
                      ovos-skill-wallpapers.openvoiceos:picture.about.intent       0.80      1.00      0.89       160
                     ovos-skill-wallpapers.openvoiceos:picture.random.intent       0.96      0.82      0.89       160
                    ovos-skill-wallpapers.openvoiceos:wallpaper.about.intent       0.94      0.99      0.97       160
                   ovos-skill-wallpapers.openvoiceos:wallpaper.random.intent       0.96      0.97      0.97       160
                              ovos-skill-weather.openvoiceos:N_days_forecast       0.60      0.60      0.60        10
                       ovos-skill-weather.openvoiceos:N_days_forecast.intent       0.90      0.94      0.92       160
                                    ovos-skill-weather.openvoiceos:condition       0.00      0.00      0.00         7
                          ovos-skill-weather.openvoiceos:current_temperature       0.00      0.00      0.00         6
                   ovos-skill-weather.openvoiceos:current_temperature.intent       0.88      0.84      0.86       124
                              ovos-skill-weather.openvoiceos:current_weather       0.33      0.25      0.29         4
                       ovos-skill-weather.openvoiceos:current_weather.intent       0.94      0.79      0.86       160
                               ovos-skill-weather.openvoiceos:daily_forecast       0.17      1.00      0.29         4
                        ovos-skill-weather.openvoiceos:daily_forecast.intent       0.88      0.61      0.72       160
                 ovos-skill-weather.openvoiceos:daily_forecast.intent.intent       0.77      0.84      0.80       159
                 ovos-skill-weather.openvoiceos:do.i.need.an.umbrella.intent       0.00      0.00      0.00         2
                                     ovos-skill-weather.openvoiceos:forecast       0.26      0.71      0.38         7
                             ovos-skill-weather.openvoiceos:high_temperature       0.33      0.33      0.33         3
                      ovos-skill-weather.openvoiceos:high_temperature.intent       0.90      0.99      0.94       160
                              ovos-skill-weather.openvoiceos:hourly_forecast       0.13      0.50      0.21         4
                       ovos-skill-weather.openvoiceos:hourly_forecast.intent       0.79      0.97      0.87       160
                           ovos-skill-weather.openvoiceos:hourly_temperature       0.20      0.11      0.14         9
                    ovos-skill-weather.openvoiceos:hourly_temperature.intent       0.95      0.98      0.97       160
                                     ovos-skill-weather.openvoiceos:humidity       0.38      1.00      0.55         3
                              ovos-skill-weather.openvoiceos:humidity.intent       0.88      0.94      0.91        48
                                     ovos-skill-weather.openvoiceos:is_clear       0.00      0.00      0.00         7
                              ovos-skill-weather.openvoiceos:is_clear.intent       0.96      0.75      0.84        95
                                       ovos-skill-weather.openvoiceos:is_fog       0.00      0.00      0.00         4
                                ovos-skill-weather.openvoiceos:is_fog.intent       0.92      0.98      0.95        87
                                      ovos-skill-weather.openvoiceos:is_snow       0.00      0.00      0.00         3
                               ovos-skill-weather.openvoiceos:is_snow.intent       0.86      0.81      0.83        67
                                    ovos-skill-weather.openvoiceos:is_stormy       1.00      0.50      0.67         4
                             ovos-skill-weather.openvoiceos:is_stormy.intent       0.91      0.91      0.91        95
                                      ovos-skill-weather.openvoiceos:is_wind       0.40      0.80      0.53         5
                               ovos-skill-weather.openvoiceos:is_wind.intent       0.93      0.99      0.96       152
                              ovos-skill-weather.openvoiceos:low_temperature       0.00      0.00      0.00         3
                       ovos-skill-weather.openvoiceos:low_temperature.intent       0.83      1.00      0.91       160
                                    ovos-skill-weather.openvoiceos:next_rain       0.00      0.00      0.00         4
                             ovos-skill-weather.openvoiceos:next_rain.intent       0.47      0.65      0.54        34
                                      ovos-skill-weather.openvoiceos:sunrise       0.20      0.12      0.15         8
                               ovos-skill-weather.openvoiceos:sunrise.intent       0.89      0.78      0.84        65
                                       ovos-skill-weather.openvoiceos:sunset       0.33      0.38      0.35         8
                                ovos-skill-weather.openvoiceos:sunset.intent       0.74      0.89      0.81        62
                             ovos-skill-weather.openvoiceos:weekend_forecast       1.00      0.20      0.33         5
                      ovos-skill-weather.openvoiceos:weekend_forecast.intent       0.81      0.99      0.89       160
                               ovos-skill-wikihow.openvoiceos:wikihow.intent       0.98      0.74      0.84        61
                               ovos-skill-wikipedia.openvoiceos:common_query       0.11      0.33      0.17         6
                                ovos-skill-wikipedia.openvoiceos:wiki.intent       0.83      0.97      0.89       116
                        ovos-skill-wikipedia.openvoiceos:wikiroulette.intent       0.76      0.98      0.86       128
                          ovos-skill-wolfie.openvoiceos:search_wolfie.intent       0.82      1.00      0.90        68
                               ovos-skill-wordnet.openvoiceos:antonym.intent       0.76      0.94      0.84        17
                            ovos-skill-wordnet.openvoiceos:definition.intent       0.81      0.47      0.60       160
                               ovos-skill-wordnet.openvoiceos:holonym.intent       0.82      0.88      0.85        16
                              ovos-skill-wordnet.openvoiceos:hypernym.intent       0.81      0.91      0.86        33
                               ovos-skill-wordnet.openvoiceos:hyponym.intent       0.79      0.94      0.86        65
                                 ovos-skill-wordnet.openvoiceos:lemma.intent       0.75      0.92      0.83        13
                        ovos-skill-wordnet.openvoiceos:search_wordnet.intent       0.78      0.82      0.80        22
                               ovos-skill-wordnet.openvoiceos:synonym.intent       0.42      0.71      0.53         7
                                                                   stop:stop       0.88      0.55      0.68        83

                                                                    accuracy                           0.82     10516
                                                                   macro avg       0.57      0.57      0.54     10516
                                                                weighted avg       0.83      0.82      0.81     10516

```

## Per-language evaluation

| Language | Samples | Accuracy | Weighted F1 | Throughput (sps) |
|---|---|---|---|---|
| ca | 2955 | 0.9140 | 0.9086 | 24322 |
| da | 527 | 0.7552 | 0.7361 | 25912 |
| de | 864 | 0.7882 | 0.7633 | 26536 |
| en | 1235 | 0.7279 | 0.7002 | 24675 |
| es | 1163 | 0.8151 | 0.7984 | 23921 |
| eu | 230 | 0.6435 | 0.6302 | 24457 |
| fr | 202 | 0.5842 | 0.5746 | 19372 |
| gl | 913 | 0.9113 | 0.8985 | 25793 |
| it | 682 | 0.8710 | 0.8602 | 22680 |
| nl | 355 | 0.6366 | 0.6303 | 23507 |
| pt | 1390 | 0.8065 | 0.7931 | 26127 |

