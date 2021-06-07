# For GPT-2 variant training and inference on team-level task
python model.py --train_file ../../data/train.json \
                --test_file ../../data/test.json \
                --save ./team_save/ \
                --pred_file ./team_save/pred.text \
                --label_file ./team_save/label.txt \
                --hard_flag 0


# For GPT-2 variant training and inference on player-level task
python model.py --train_file ../../data/train.json \
                --test_file ../../data/test.json \
                --save ./player_save/ \
                --pred_file ./player_save/pred.text \
                --label_file ./player_save/label.txt \
                --hard_flag 1

# For both team-level and player-level evaluation
python evaluate.py

                
