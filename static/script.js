$('#file-upload').submit(function(event) {
    event.preventDefault();

    $('#structify-output').html(`
        <div class="progress">
            <div
                class="progress-bar progress-bar-striped active"
                role="progressbar"
                aria-valuenow="45"
                aria-valuemin="0"
                aria-valuemax="100"
                style="width: 100%"
            >
            </div>
        </div>
    `);

    $('#submit').attr('disabled', 'disabled');

    const data = new FormData($(this)[0]);
    $.ajax({
        url: 'upload',
        data: data,
        type:'POST',
        contentType: false,
        processData: false,
        success: function(data) {
            $('#structify-output').empty();

            // Display image and audio file
            const image = document.createElement('img');
            $(image).attr('src', data.image);
            $(image).attr('id', 'structify-output-image');

            const audio = document.createElement('audio');
            $(audio).attr('controls', 'controls');
            $(audio).attr('id', 'structify-output-audio');
            const audio_source = document.createElement('source');
            $(audio_source).attr('src', data.audio);
            $(audio_source).attr('type', 'audio/wav');
            $(audio).append(audio_source);

            $('#structify-output').append(image, audio);
            $('#submit').prop('disabled', false);
        }
    });
});
