echo Downloading Weights!
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1E8kx197WNIZydk_sKBxb7HjRYrJVSenr' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1E8kx197WNIZydk_sKBxb7HjRYrJVSenr" -O data.zip && rm -rf /tmp/cookies.txt

unzip data.zip
rm data.zip

echo Extracting Training Data!

mv triple_worms_3/triple_output/Alive .
mv triple_worms_3/triple_output/Dead .

rm -rf triple_worms_3

echo Done!