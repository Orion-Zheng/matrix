from decoupled_the_matrix import the_matrix
video_prompt = "The video shows a white car driving on a country road on a sunny day. The car comes from the back of the scene, moving forward along the road, with open fields and distant hills surrounding it. As the car moves, the vegetation on both sides of the road and distant buildings can be seen. The entire video records the car's journey through the natural environment using a follow-shot technique."
the_matrix_generator = the_matrix(generation_model_path="/matrix_ckpts/stage2", streaming_model_path="/matrix_ckpts/stage3")
the_matrix_generator.generate(
    prompt=video_prompt,
    length=8,
    output_folder="./",
    control_signal="D,D,D,D,DL,DL,DL,DL,DL,DL,D,D,D,D,D,D,D"
)