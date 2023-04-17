# Function for turning lists into scores. Added manual measures because of how chronotonic distance function works. 

from music21 import *
import music21
import math
from tqdm import *
import tqdm
import pickle 
from fractions import Fraction
import random
import os
import numpy as np
import pandas as pd
import collections

UNIT = 1/4 #We express duration as fractions of a quarter note. Use UNIT = 1 to express durations as fractions of whole-notes

def percussion_hit(duration, pitch = -2):
    #Create Note object for percussion hits (default pitch is C4)
    if pitch == -2:
        print('missing pitches')
    if pitch == 0:
        pitch_to_add = 'C'
    if pitch == 1:
        pitch_to_add = 'C#'
    if pitch == 2:
        pitch_to_add = 'D'
    if pitch == 3:
        pitch_to_add = 'D#'
    if pitch == 4:
        pitch_to_add = 'E'
    if pitch == 5:
        pitch_to_add = 'F'
    if pitch == 6:
        pitch_to_add = 'F#'
    if pitch == 7:
        pitch_to_add = 'G'
    if pitch == 8:
        pitch_to_add = 'G#'
    if pitch == 9:
        pitch_to_add = 'A'
    if pitch == 10:
        pitch_to_add = 'A#'
    if pitch == 11:
        pitch_to_add = 'B'    
    return note.Note(pitch_to_add, quarterLength = duration*(4*UNIT))

def create_percussion(time_sig = None):
    """ Creates a container with or without time signature to which
    events can be appended.

    Parameters
    ----------
    time_sig : str, optional
        By default, no time signature is specified, so that a single measure of unspecified capacity is generated.
        If you do specify a time signature a stream with the given time signature is created instead.

    Returns
    -------
    music21.stream.Stream
        If time_sig is None, the return value is a Measure which is a subclass of Stream.
    """
    if time_sig == None:
        drumPart = stream.Measure()
    else:
        drumPart = stream.Stream()
        drumPart.timeSignature = meter.TimeSignature(time_sig)
    
    # drumPart.insert(1, instrument.Woodblock()) #assign woosblock timbre
    return drumPart

def append_event(duration, old_score, rest = False, pitch = -2):
    """ Appends a rest or percussion_hit of the given duration to an existing stream.

    Parameters
    ----------
    duration : float or Fraction
        Duration of the new event.
    stream_object : music21.stream.Stream
        Stream object to which append the new event to.
    rest : bool, optional
        By default, the appended event has the pitch ``pitch``.
        Pass True to append a rest instead.
    pitch : str, optional
        Pitch of the appended event, matters only for display. Disregarded if ``rest`` is True.

    Returns
    -------
    None
    """
    new_score = old_score
    if rest:
        new_score.getElementsByClass(stream.Measure)[-1].append(note.Rest(quarterLength = duration*(4*UNIT)))
    else:
        new_score.getElementsByClass(stream.Measure)[-1].append(percussion_hit(duration, pitch))
    return new_score

def score_from_sequence(events, time_sig = None, score=None):
    """ Generate rhythmic stream from a list of durations. Rests are indicated by specifying a duration as a string.

    Parameters
    ----------
    durations : list of {float, Fraction, str}
        Sequence of durations. Floats or Fractions passed as strings are interpreted as rests.
    time_sig : str, optional
        By default, no time signature is specified, so that a single measure of unspecified capacity is generated.
        If you do specify a time signature a stream with the given time signature is created instead.
    pitch : str, optional
        Matters only for display.
    rhythm : music21.stream.Stream, optional
        If you want to append the rhythm to an existing Stream, pass this Stream. Otherwise a new one will be created.

    Returns
    -------
    music21.stream.Stream
        The rhythm reflecting the given list of durations.
    """
    if score is None:
        # pass an existing stream 'rhythm' to append the durations, otherwise a new one will be created
        score = create_percussion(time_sig = time_sig)
        score.insert(1,stream.Measure(number=1))
        #score.getElementsByClass(stream.Measure)[-1].append(music21.key.Key('a'))
        meas = score.getElementsByClass(stream.Measure)[-1].measureNumber
        running_onset = 1
    for eve in events:
        dur = eve[0]
        pitch = eve[1]
        running_onset = running_onset + float(dur)
        is_rest = False
        if dur != 0:
            if pitch == -1:
                is_rest = True
            dist_above_measure = running_onset - (((meas-1)*3)+1)
            if dist_above_measure > 3:
                score.insert((((meas-1)*3)+1)+3,stream.Measure(number=meas + 1))
                meas += 1
            score = append_event(dur, score, is_rest, pitch)
    return score

def import_slangpolskas(xml_file_location, keepScoresWithChords = False):
    scores_xml = os.listdir(xml_file_location)
    slangpolska_scores = {}
    excluded = 0
    for si in trange(len(scores_xml)):
        if scores_xml[si] != 'conversion.log':
            tmp = converter.parse(xml_file_location + scores_xml[si])
            tmp_meters = tmp.recurse().getElementsByClass(meter.TimeSignature)
            if len(tmp_meters) != 1 or tmp_meters[0].ratioString != '3/4':
                print('excluding ' + scores_xml[si] + '. ' + str(len(tmp_meters)) + ' meter(s), first is ' + tmp_meters[0].ratioString)
                excluded += 1
            elif tmp.recurse().notesAndRests[-1].measureNumber < 8:
                print('excluding ' + scores_xml[si] + '. Only ' + str(tmp.recurse().notesAndRests[-1].measureNumber) + ' measures.')
                excluded += 1
            else:
                chord_found = 0
                if keepScoresWithChords == False:
                    for event in tmp.recurse().notes:
                        if event.measureNumber > 8:
                            # pass notes test, proceed
                            break
                        try:
                            event.pitchedCommonName
                            chord_found = 1
                        except:
                            pass
                if chord_found == 1:
                    print('excluding ' + scores_xml[si] + '. A chord was found in the first 8 measures.')
                    excluded += 1
                else:
                    slangpolska_scores[scores_xml[si]] = tmp

    print('Imported ' + str(len(slangpolska_scores)) + ' scores into corpus. Excluded ' + str(excluded) + ' due to inconsistent meter, polyphonic setup, less than 8 measures, or chord presence in the first 8 measures.')
    return slangpolska_scores

# Distance function in ipynb was first pass for visualization. This one is faster and equivalent.
def chronotonic_sequence_create_and_store(full_chronotonic_sequence_holder, stream_base_score, stream_name):
    if stream_name in full_chronotonic_sequence_holder:
        pass
    else:
        onsets = []
        heights = []
        final_measure = stream_base_score.recurse().notesAndRests[-1].measureNumber
        for n_or_r in stream_base_score.recurse().notesAndRests:
            #print(n_or_r)
            onsets.append(n_or_r.beat + ((3 * (n_or_r.measureNumber-1))))
            if isinstance(n_or_r, (music21.note.Note,music21.chord.Chord)):
                heights.append(n_or_r.duration.quarterLength)
            else:
                heights.append(float(0))
        x_for_interp = []
        y_for_interp = []
        for i,x in enumerate(onsets):
            if i != 0:
                x_for_interp.append(float(onsets[i-1]))
                y_for_interp.append(float(heights[i-1]))
                x_for_interp.append(float(onsets[i]-.00000000001))
                y_for_interp.append(float(heights[i-1]))
        x_for_interp.append(float(onsets[-1]))
        y_for_interp.append(float(heights[-1]))
        xvals = (np.arange(0,(final_measure*(1/.01))*3+1)*.01+1).tolist()
        #xvals = np.linspace(0, final_measure, final_measure * 1250)
        yinterp = np.interp(xvals, x_for_interp, y_for_interp)
        full_chronotonic_sequence_holder[stream_name] = (xvals, yinterp)
    return full_chronotonic_sequence_holder

def chronotonic_comparison_rand8bars_alt(full_chronotonic_sequence_holder, score1, score2, name1, name2, subset_start = -1):
    #print("trying " + name1 + " and " + name2)
    if name1 not in full_chronotonic_sequence_holder:
        #print("chronotizing " + name1)
        full_chronotonic_sequence_holder = chronotonic_sequence_create_and_store(full_chronotonic_sequence_holder, score1, name1)
    if name2 not in full_chronotonic_sequence_holder:
        #print("chronotizing " + name2)
        full_chronotonic_sequence_holder = chronotonic_sequence_create_and_store(full_chronotonic_sequence_holder, score2, name2)
    
    # Random odd number small enough to pull 8 bars from both scores
    final_measure_1 = ((full_chronotonic_sequence_holder[name1][0][-1]-1)/3)
    #print(final_measure_1, full_chronotonic_sequence_holder[name1][-5:])
    final_measure_2 = ((full_chronotonic_sequence_holder[name2][0][-1]-1)/3)
    #print(final_measure_2)
    final_measure_for_subset = final_measure_1
    if final_measure_2 < final_measure_for_subset:
        final_measure_for_subset = final_measure_2
    if (final_measure_for_subset < 8) or (final_measure_for_subset < (subset_start + 7)):
        print("ONE OF THE SCORES IS TOO SHORT")
        return "ONE OF THE SCORES IS TOO SHORT"
    if subset_start == -1:
        while True:
            subset_start = random.randint(1, (final_measure_for_subset - 8)+1)
            if subset_start%2 != 0:
                break

    #print(subset_start)
    linspaced_score1 = pd.DataFrame(list(zip(full_chronotonic_sequence_holder[name1][0], full_chronotonic_sequence_holder[name1][1])), columns=['score1_xvals', 'score1_yvals'])
    linspaced_score2 = pd.DataFrame(list(zip(full_chronotonic_sequence_holder[name2][0], full_chronotonic_sequence_holder[name2][1])), columns=['score2_xvals', 'score2_yvals'])

    # Filter linspaced_sequences to 8bar subset
    tmp_indices_score1 = np.where(np.logical_and(linspaced_score1['score1_xvals'] >= ((subset_start-1)*3)+1,linspaced_score1['score1_xvals'] < (((subset_start+8)-1)*3) + 1), True, False)
    tmp_indices_score2 = np.where(np.logical_and(linspaced_score2['score2_xvals'] >= ((subset_start-1)*3)+1,linspaced_score2['score2_xvals'] < (((subset_start+8)-1)*3) + 1), True, False)
    linsubs_xvals_score1 = linspaced_score1['score1_xvals'].loc[tmp_indices_score1]
    linsubs_yvals_score1 = linspaced_score1['score1_yvals'].loc[tmp_indices_score1]

    linsubs_xvals_score2 = linspaced_score2['score2_xvals'].loc[tmp_indices_score2]
    linsubs_yvals_score2 = linspaced_score2['score2_yvals'].loc[tmp_indices_score2]

    x_diff = linsubs_xvals_score1 - linsubs_xvals_score2
    y_diff = abs(linsubs_yvals_score1 - linsubs_yvals_score2)

    #plt.figure()
    #plt.plot(linsubs_xvals_score1, linsubs_yvals_score1, 'blue', alpha = 0.9)
    #plt.plot(linsubs_xvals_score2, linsubs_yvals_score2, 'red', alpha = 0.9)
    #plt.show()

    #print(linsubs_xvals_score1 - 7)
    #print(len(linsubs_xvals_score1), len(y_diff))
    #if len(linsubs_xvals_score1) == 0:
    #    print(subset_start)
    #    print(linspaced_score1['score1_xvals'])
    #    print(linsubs_xvals_score1)
    return full_chronotonic_sequence_holder, np.trapz(y_diff, linsubs_xvals_score1)

# function to create string of concatenated pitches from first 8 measures of score
def pitch_sequence(score):
    pitch_sequence = ''
    for i in score.recurse().notes:
        if i.measureNumber > 8:
            break
        try:
            print(score_name, i.pitchedCommonName, i.measureNumber)
        except:
            pass

        pitch_to_add = i.name
        if pitch_to_add in ["C#","D-"]:
            pitch_to_add = "d"
        if pitch_to_add in ["D#","E-"]:
            pitch_to_add = "e"
        if pitch_to_add in ["F#","G-"]:
            pitch_to_add = "g"
        if pitch_to_add in ["G#","A-"]:
            pitch_to_add = "a"
        if pitch_to_add in ["A#","B-"]:
            pitch_to_add = "b"
        pitch_sequence = pitch_sequence + pitch_to_add
    return pitch_sequence

# Weighted Levenshtein - adapted from https://github.com/infoscout/weighted-levenshtein, which was written for Python 2
def levenshtein_wrapper(str1, str2, insert_costs=None, delete_costs=None, substitute_costs=None):
    """
    Calculates the Levenshtein distance between str1 and str2,
    provided the costs of inserting, deleting, and substituting characters.
    The costs default to 1 if not provided.
    :param str str1: first string
    :param str str2: second string
    :param np.ndarray insert_costs: a numpy array of np.float64 of length 128 (0..127),
        where insert_costs[i] is the cost of inserting ASCII character i
    :param np.ndarray delete_costs: a numpy array of np.float64 of length 128 (0..127),
        where delete_costs[i] is the cost of deleting ASCII character i
    :param np.ndarray substitute_costs: a 2D numpy array of np.float64 of dimensions (128, 128),
        where substitute_costs[i, j] is the cost of substituting ASCII character i with
        ASCII character j
    """
    if insert_costs is None:
        insert_costs = np.ones(128)
    if delete_costs is None:
        delete_costs = np.ones(128)
    if substitute_costs is None:
        substitute_costs = np.ones([128,128]) 

    return levenshtein(
        str1, len(str1),
        str2, len(str2),
        insert_costs,
        delete_costs,
        substitute_costs
    )
def levenshtein(str1, len1, str2, len2, insert_costs, delete_costs, substitute_costs):
    """
    https://en.wikipedia.org/wiki/Wagner%E2%80%93Fischer_algorithm
    """
    d = np.zeros([len1+1, len2+1])
    for i in range(1,len1+1):
        char_i = str1[i-1]
        d[i,0] = d[i-1,0] + delete_costs[ord(char_i)]
    for j in range(1,len2+1):
        char_j = str2[j-1]
        d[0,j] = d[0,j-1] + insert_costs[ord(char_j)]

    for i in range(1, len1+1):
        char_i = str1[i-1]
        for j in range(1, len2+1):
            char_j = str2[j-1]
            if char_i == char_j:  # match
                d[i,j] = d[i-1,j-1]
            else:
                d[i,j] = min(
                    d[i-1,j] + delete_costs[ord(char_i)],
                    d[i,j-1] + insert_costs[ord(char_j)],
                    d[i-1,j-1] + substitute_costs[ord(char_i), ord(char_j)]
                )
    ret_val = d[len1, len2]
    return ret_val

# Function for comparing scores - can compare pitch (levenshtein distance) or rhythm (chronontonic difference)
def compare(set1, set2, set1_moniker, set2_moniker, comparison_type, full_chronotonic_sequence_holder = None):
    
    num_set1 = len(set1)
    set1_to_set2_distances = {}

    if comparison_type == "pitch":
        with open('levenshtein_costs/insert_costs.pkl', 'rb') as f:
            insert_costs = pickle.load(f)
        with open('levenshtein_costs/delete_costs.pkl', 'rb') as f:
            delete_costs = pickle.load(f)
        with open('levenshtein_costs/substitute_costs.pkl', 'rb') as f:
            substitute_costs = pickle.load(f)

    if comparison_type == "rhythm" and full_chronotonic_sequence_holder == None:
        print('here')
        full_chronotonic_sequence_holder = {}
    
    for i, score_i in enumerate(tqdm.tqdm(set1)):
        # print(i)
        if type(set1) is dict:
            set1_score = set1[score_i]
            set1_name = score_i
        else:
            if comparison_type == "pitch":
                set1_score = score_i
            elif comparison_type == "rhythm":
                set1_score = stream.Score(score_i)
            set1_name = set1_moniker + '_' + str(i)
            
        set1_to_set2_distances[set1_name] = {}

        for j, score_j in enumerate(set2):
        
            if type(set2) is dict:
                set2_score = set2[score_j]
                set2_name = score_j
            else:
                if comparison_type == "pitch":
                    set2_score = score_j
                elif comparison_type == "rhythm":
                    set2_score = stream.Score(score_j)
                set2_name = set2_moniker + '_' + str(j)

            if set1_to_set2_distances.get(set2_name, {}).get(set1_name) is not None:
                set1_to_set2_distances[set1_name][set2_name] = set1_to_set2_distances[set2_name][set1_name]
            else:
                if comparison_type == "pitch":
                    set1_to_set2_distances[set1_name][set2_name] = levenshtein_wrapper(pitch_sequence(set1_score),pitch_sequence(set2_score),insert_costs, delete_costs, substitute_costs)
                elif comparison_type == "rhythm":
                    # the _holder object saves full chronotonic sequences as a tuple: (x_vals, y_vals). Interpoliation provides 1250 points per measure.
                    # to give an idea of quality of approximation, an 8-bar difference that should evaluate to 9 evaluates to 8.997599759975998 
                    full_chronotonic_sequence_holder, set1_to_set2_distances[set1_name][set2_name] = chronotonic_comparison_rand8bars_alt(full_chronotonic_sequence_holder, set1_score, set2_score, set1_name, set2_name, subset_start=1)
                    #print(set1_to_set2_distances[set1_name][set2_name])
        
        if (i%25 == 0) or (i >= num_set1 - 1):
            with open('all_distances/' + set1_moniker + '_to_' + set2_moniker + '_' + comparison_type + '_distances' + '.pkl', 'wb') as f:
                pickle.dump(set1_to_set2_distances, f)
        
    if comparison_type == "rhythm":
        return full_chronotonic_sequence_holder